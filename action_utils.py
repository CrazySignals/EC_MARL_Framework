import numpy as np
import torch
from torch.autograd import Variable

def parse_action_args(args):
    if args.num_actions[0] > 0:
        # environment takes discrete action
        args.continuous = False
        # assert args.dim_actions == 1
        # support multi action
        args.naction_heads = [int(args.num_actions[i]) for i in range(args.dim_actions)]
    else:
        # environment takes continuous action
        actions_heads = args.nactions.split(':')
        if len(actions_heads) == 1 and int(actions_heads[0]) == 1:
            args.continuous = True
        elif len(actions_heads) == 1 and int(actions_heads[0]) > 1:
            args.continuous = False
            args.naction_heads = [int(actions_heads[0]) for _ in range(args.dim_actions)]
        elif len(actions_heads) > 1:
            args.continuous = False
            args.naction_heads = [int(i) for i in actions_heads]
        else:
            raise RuntimeError("--nactions wrong format!")


def select_action(args, action_out):
    # 获取设备
    device = action_out[0][0].device if not args.continuous else action_out[0].device
    
    if args.continuous:
        action_mean, _, action_std = action_out
        action = torch.normal(action_mean, action_std)
        return action.detach()
    else:
        log_p_a = action_out
        p_a = [[z.exp() for z in x] for x in log_p_a]
        ret = []
        for agent_p in p_a:
            agent_actions = []
            for action_probs in agent_p:
                # 确保在同一设备上进行操作
                action = torch.multinomial(action_probs, 1).detach()
                agent_actions.append(action)
            ret.append(torch.stack(agent_actions))
        return torch.stack(ret)

def translate_action(args, env, action):
    if args.num_actions[0] > 0:
        # environment takes discrete action
        # 将 tensor 转换为 numpy，需要先移到 CPU
        if isinstance(action[0], torch.Tensor):
            action = [x.squeeze().cpu().numpy() for x in action]
        else:
            action = [x.squeeze().data.numpy() for x in action]
        actual = action
        if not hasattr(actual, '__len__'):
            actual = [actual] * args.nagents
        return action, actual
    else:
        if args.continuous:
            # 确保 action 是 CPU 上的 numpy 数组
            if isinstance(action, torch.Tensor):
                action = action.cpu().data[0].numpy()
            else:
                action = action.data[0].numpy()
                
            cp_action = action.copy()
            # clip and scale action to correct range
            for i in range(len(action)):
                low = env.action_space.low[i]
                high = env.action_space.high[i]
                cp_action[i] = cp_action[i] * args.action_scale
                cp_action[i] = max(-1.0, min(cp_action[i], 1.0))
                cp_action[i] = 0.5 * (cp_action[i] + 1.0) * (high - low) + low
            return action, cp_action
        else:
            actual = np.zeros(len(action))
            for i in range(len(action)):
                low = env.action_space.low[i]
                high = env.action_space.high[i]
                # 确保在 CPU 上处理
                if isinstance(action[i], torch.Tensor):
                    actual[i] = action[i].cpu().data.squeeze()[0] * (high - low) / (args.naction_heads[i] - 1) + low
                else:
                    actual[i] = action[i].data.squeeze()[0] * (high - low) / (args.naction_heads[i] - 1) + low
                    
                if not hasattr(actual, '__len__'):
                    actual = [actual] * args.nagents
                    
            # 确保 action 也在 CPU 上
            if isinstance(action[0], torch.Tensor):
                action = [x.squeeze().cpu().data[0] for x in action]
            else:
                action = [x.squeeze().data[0] for x in action]
                
            return action, actual

def epsilon_greedy_select_action(args, action_out, epsilon=0.1):
    """
    Epsilon-greedy action selection for discrete actions.
    
    Args:
        args: Arguments containing action parameters
        action_out: Output from the policy network (log probabilities)
        epsilon: Probability of choosing a random action
    
    Returns:
        Selected actions tensor
    """
    # 获取当前设备
    device = action_out[0][0].device if not args.continuous else action_out[0].device
    
    if args.continuous:
        # For continuous actions, we can add noise to the mean
        action_mean, action_std = action_out
        if np.random.random() < epsilon:
            # Explore: increase the standard deviation
            action_std = action_std * 2.0  
        action = torch.normal(action_mean, action_std)
        return action.detach()
    else:
        log_p_a = action_out
        p_a = [[z.exp() for z in x] for x in log_p_a]
        
        ret = []
        for agent_p in p_a:
            agent_actions = []
            for action_probs in agent_p:
                if np.random.random() < epsilon:
                    # Explore: random action - 确保在同一设备上
                    uniform_probs = torch.ones_like(action_probs, device=device) / action_probs.size(0)
                    action = torch.multinomial(uniform_probs, 1)
                else:
                    # Exploit: greedy action
                    action = torch.argmax(action_probs).unsqueeze(0)
                agent_actions.append(action.detach())
            ret.append(torch.stack(agent_actions))
        
        return torch.stack(ret)

def ucb_select_action(args, action_out, action_counts, t, c=2.0):
    """
    Upper Confidence Bound action selection.
    
    Args:
        args: Arguments containing action parameters
        action_out: Output from the policy network (log probabilities)
        action_counts: Count of how many times each action has been taken
        t: Current time step
        c: Exploration parameter
    
    Returns:
        Selected actions tensor
    """
    # 获取当前设备
    device = action_out[0][0].device if not args.continuous else action_out[0].device
    
    if args.continuous:
        # UCB is typically used for discrete actions, but we can
        # adapt it by adding uncertainty-based noise
        action_mean, action_std = action_out
        # Add UCB-inspired bonus to the standard deviation
        # 确保 t 也在正确的设备上
        t_tensor = torch.tensor(t + 1, device=device, dtype=torch.float)
        # 确保 action_counts 在正确的设备上
        if isinstance(action_counts, torch.Tensor) and action_counts.device != device:
            action_counts = action_counts.to(device)
            
        bonus = c * torch.sqrt(torch.log(t_tensor) / (action_counts + 1))
        adjusted_std = action_std + bonus.unsqueeze(-1)
        action = torch.normal(action_mean, adjusted_std)
        return action.detach()
    else:
        log_p_a = action_out
        p_a = [[z.exp() for z in x] for x in log_p_a]
        
        # 确保 t 在正确的设备上
        t_tensor = torch.tensor(t + 1, device=device, dtype=torch.float)
        
        ret = []
        for agent_idx, agent_p in enumerate(p_a):
            agent_actions = []
            for action_idx, action_probs in enumerate(agent_p):
                # 确保 action_counts 在正确的设备上
                if action_counts[agent_idx][action_idx].device != device:
                    action_counts[agent_idx][action_idx] = action_counts[agent_idx][action_idx].to(device)
                
                # Calculate UCB scores
                q_values = action_probs  # Using probabilities as Q-values
                counts = action_counts[agent_idx][action_idx]
                ucb_scores = q_values + c * torch.sqrt(
                    torch.log(t_tensor) / (counts + 1)
                )
                # Select action with highest UCB score
                action = torch.argmax(ucb_scores).unsqueeze(0)
                
                # Update counts for the selected action
                action_counts[agent_idx][action_idx][action] += 1
                
                agent_actions.append(action.detach())
            ret.append(torch.stack(agent_actions))
        
        return torch.stack(ret)

def thompson_sampling_select_action(args, action_out, alpha, beta):
    """
    Thompson Sampling for action selection.
    
    Args:
        args: Arguments containing action parameters
        action_out: Output from the policy network (log probabilities)
        alpha: Success parameter for Beta distribution
        beta: Failure parameter for Beta distribution
    
    Returns:
        Selected actions tensor
    """
    # 获取当前设备
    device = action_out[0][0].device if not args.continuous else action_out[0].device
    
    if args.continuous:
        # For continuous actions, sample from a distribution informed by Thompson sampling
        action_mean, action_std = action_out
        
        # 确保 alpha 和 beta 在 CPU 上进行采样
        alpha_cpu = alpha.cpu().numpy() if isinstance(alpha, torch.Tensor) else alpha
        beta_cpu = beta.cpu().numpy() if isinstance(beta, torch.Tensor) else beta
        
        # Sample variance multiplier from a distribution
        variance_multiplier = torch.tensor(
            np.random.gamma(alpha_cpu, 1.0 / beta_cpu)
        ).float().to(device)
        adjusted_std = action_std * torch.sqrt(variance_multiplier).unsqueeze(-1)
        action = torch.normal(action_mean, adjusted_std)
        return action.detach()
    else:
        log_p_a = action_out
        p_a = [[z.exp() for z in x] for x in log_p_a]
        
        ret = []
        for agent_idx, agent_p in enumerate(p_a):
            agent_actions = []
            for action_idx, action_probs in enumerate(agent_p):
                # 确保 alpha 和 beta 在 CPU 上进行采样
                alpha_cpu = alpha[agent_idx][action_idx].cpu().numpy() if isinstance(alpha[agent_idx][action_idx], torch.Tensor) else alpha[agent_idx][action_idx]
                beta_cpu = beta[agent_idx][action_idx].cpu().numpy() if isinstance(beta[agent_idx][action_idx], torch.Tensor) else beta[agent_idx][action_idx]
                
                # Sample from Beta distribution for each action
                beta_samples = torch.tensor(
                    np.random.beta(alpha_cpu, beta_cpu)
                ).float().to(device)
                
                # Select action with highest sample
                action = torch.argmax(beta_samples).unsqueeze(0)
                agent_actions.append(action.detach())
            ret.append(torch.stack(agent_actions))
        
        return torch.stack(ret)

def boltzmann_select_action(args, action_out, temperature=1.0):
    """
    Boltzmann exploration for action selection.
    
    Args:
        args: Arguments containing action parameters
        action_out: Output from the policy network (log probabilities)
        temperature: Temperature parameter for softmax
    
    Returns:
        Selected actions tensor
    """
    # 获取当前设备
    device = action_out[0][0].device if not args.continuous else action_out[0].device
    
    if args.continuous:
        action_mean, action_std = action_out
        # Adjust the standard deviation based on temperature
        adjusted_std = action_std * temperature
        action = torch.normal(action_mean, adjusted_std)
        return action.detach()
    else:
        log_p_a = action_out
        p_a = []
        
        for agent_log_probs in log_p_a:
            agent_p = []
            for log_probs in agent_log_probs:
                # Apply temperature scaling to logits (convert log_probs back to logits)
                scaled_logits = log_probs / temperature
                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
                agent_p.append(probs)
            p_a.append(agent_p)
        
        ret = []
        for agent_p in p_a:
            agent_actions = []
            for action_probs in agent_p:
                # 在同一设备上进行采样
                action = torch.multinomial(action_probs, 1).detach()
                agent_actions.append(action)
            ret.append(torch.stack(agent_actions))
        
        return torch.stack(ret)

# 添加这些辅助函数用于计算多项分布的对数密度，支持GPU

def multinomials_log_density(actions, log_probs):
    """
    计算多项分布的对数密度。
    
    Args:
        actions: 选择的动作 [batch_size, dim_actions]
        log_probs: 动作的对数概率 [list of [batch_size, num_actions]]
        
    Returns:
        对数密度 [batch_size, 1]
    """
    # 获取当前设备
    device = log_probs[0].device
    
    result = torch.zeros(actions.size(0), device=device)
    for i in range(len(log_probs)):
        indices = actions[:, i].long()
        
        # 确保 indices 有正确的形状
        if indices.dim() == 2:
            indices = indices.squeeze(1)
        
        # 获取每个批次中选中动作的对数概率
        result += log_probs[i].gather(1, indices.unsqueeze(1)).squeeze()
    
    return result.unsqueeze(1)

def multinomials_log_densities(actions, log_probs):
    """
    计算每个动作头的对数密度。
    
    Args:
        actions: 选择的动作 [batch_size, dim_actions]
        log_probs: 动作的对数概率 [list of [batch_size, num_actions]]
        
    Returns:
        每个动作头的对数密度 [batch_size, dim_actions]
    """
    # 获取当前设备
    device = log_probs[0].device
    
    densities = []
    for i in range(len(log_probs)):
        indices = actions[:, i].long()
        
        # 确保 indices 有正确的形状
        if indices.dim() == 2:
            indices = indices.squeeze(1)
        
        # 计算对数密度并添加到列表中
        densities.append(log_probs[i].gather(1, indices.unsqueeze(1)))
    
    return torch.cat(densities, dim=1)