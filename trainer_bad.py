from collections import namedtuple
from inspect import getfullargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utils import *
from action_utils import *
import itertools

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state', 'reward', 'misc'))

def get_temperature(epoch, max_epochs, initial_temp=2.0, final_temp=0.5):
    """Linearly anneal temperature from initial to final value over training."""
    return initial_temp - (initial_temp - final_temp) * (epoch / max_epochs)

def initialize_action_counts(args):
    """Initialize action counts for UCB."""
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    if args.continuous:
        return torch.zeros(args.nagents, args.dim_actions, device=device)
    else:
        return [[torch.zeros(n_actions, device=device) for n_actions in args.naction_heads] for _ in range(args.nagents)]

def initialize_thompson_params(args):
    """Initialize alpha and beta parameters for Thompson sampling."""
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    if args.continuous:
        # For continuous actions, we'll use a global alpha and beta for variance scaling
        alpha = torch.ones(args.nagents, args.dim_actions, device=device)
        beta = torch.ones(args.nagents, args.dim_actions, device=device)
    else:
        # For discrete actions, we need alpha and beta for each action
        alpha = [[torch.ones(n_actions, device=device) for n_actions in args.naction_heads] for _ in range(args.nagents)]
        beta = [[torch.ones(n_actions, device=device) for n_actions in args.naction_heads] for _ in range(args.nagents)]
    return alpha, beta

class Trainer(object):
    def __init__(self, args, policy_net, env):
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.display = False
        self.last_step = False
        
        # 设置设备
        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        
        # 确保策略网络在正确的设备上
        self.policy_net = self.policy_net.to(self.device)
        
        self.optimizer = optim.RMSprop(policy_net.parameters(),
            lr = args.lrate, alpha=0.97, eps=1e-6)
        self.params = [p for p in self.policy_net.parameters()]
    
    def get_episode(self, epoch):
        episode = []
        reset_args = getfullargspec(self.env.reset).args
        if 'epoch' in reset_args:
            state = self.env.reset(epoch)
        else:
            state = self.env.reset()
        should_display = self.display and self.last_step

        if should_display:
            self.env.display()
        stat = dict()
        info = dict()
        switch_t = -1

        # 将初始隐藏状态移到GPU
        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size, device=self.device)

        for t in range(self.args.max_steps):
            misc = dict()

            if t == 0:
                prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])
                # 确保隐藏状态在正确的设备上
                if isinstance(prev_hid, tuple):
                    prev_hid = (prev_hid[0].to(self.device), prev_hid[1].to(self.device))
                else:
                    prev_hid = prev_hid.to(self.device)

            # 确保状态张量在GPU上
            state_tensor = torch.from_numpy(state).to(self.device)
            x = [state_tensor, prev_hid]
            action_out, value, prev_hid = self.policy_net(x, info)

            if (t + 1) % self.args.detach_gap == 0:
                if isinstance(prev_hid, tuple):
                    prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                else:
                    prev_hid = prev_hid.detach()

            # 选择动作，确保动作选择考虑了GPU
            if self.args.action_sampling == 'epsilon_greedy':
                action = epsilon_greedy_select_action(self.args, action_out, self.args.epsilon)
            elif self.args.action_sampling == 'ucb':
                # We need to maintain action counts for UCB
                if 'action_counts' not in info:
                    info['action_counts'] = initialize_action_counts(self.args)
                    info['timestep'] = 0
                info['timestep'] += 1
                action = ucb_select_action(self.args, action_out, info['action_counts'], info['timestep'], self.args.ucb_c)
            elif self.args.action_sampling == 'thompson':
                # We need to maintain alpha and beta for Thompson sampling
                if 'thompson_alpha' not in info:
                    info['thompson_alpha'], info['thompson_beta'] = initialize_thompson_params(self.args)
                action = thompson_sampling_select_action(self.args, action_out, info['thompson_alpha'], info['thompson_beta'])
            elif self.args.action_sampling == 'boltzmann':
                current_temp = get_temperature(epoch, self.args.num_epochs)
                action = boltzmann_select_action(self.args, action_out, current_temp)
            else:
                action = select_action(self.args, action_out)
            
            # 在将动作传递给环境之前，确保它是CPU上的numpy数组
            if isinstance(action, torch.Tensor):
                action_cpu = action.cpu()
            else:
                action_cpu = action
                
            action, actual = translate_action(self.args, self.env, action_cpu)
            
            # 确保 actual 是一个列表或数组
            if isinstance(actual, torch.Tensor):
                # 如果是张量，先转换为 numpy 数组
                actual = actual.cpu().numpy()
                
            # 检查 actual 是否为标量或 0 维数组
            if np.isscalar(actual) or (isinstance(actual, np.ndarray) and actual.ndim == 0):
                actual = np.array([actual])  # 转换为 1 维数组
            elif isinstance(actual, np.ndarray) and actual.ndim == 1 and len(actual) == 1:
                # 如果是只有一个元素的 1 维数组，确保它是二维的 [[action]]
                actual = np.array([actual])
                
            next_state, reward, done, info = self.env.step(actual)

            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)

            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

            done = done or t == self.args.max_steps - 1

            episode_mask = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)

            if done:
                episode_mask = np.zeros(reward.shape)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)

            if should_display:
                self.env.display()

            # 保存经验，保持原始numpy格式以节省内存
            trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc)
            episode.append(trans)
            state = next_state
            if done:
                break
                
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']

        if hasattr(self.env, 'reward_terminal'):
            reward = self.env.reward_terminal()

            episode[-1] = episode[-1]._replace(reward = episode[-1].reward + reward)
            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)
        return (episode, stat)

    def compute_grad(self, batch):
        stat = dict()
        # num_actions: number of discrete actions in the action space
        num_actions = self.args.num_actions
        # dim_actions: number of action heads
        dim_actions = self.args.dim_actions

        n = self.args.nagents
        batch_size = len(batch.state)

        # 将所有张量移动到GPU
        # rewards: [batch_size * n]
        rewards = torch.tensor(batch.reward, device=self.device)
        # episode_mask: [batch_size * n]
        episode_masks = torch.tensor(batch.episode_mask, device=self.device)
        # episode_mini_mask: [batch_size * n]
        episode_mini_masks = torch.tensor(batch.episode_mini_mask, device=self.device)
        
        # 处理动作张量
        if isinstance(batch.action[0], torch.Tensor):
            actions = torch.cat([a.to(self.device) for a in batch.action])
        else:
            actions = torch.tensor(batch.action, device=self.device)
            
        # actions: [batch_size * n * dim_actions] 
        actions = actions.transpose(1, 2).view(-1, n, dim_actions)

        # 处理值函数张量
        if isinstance(batch.value[0], torch.Tensor):
            values = torch.cat([v.to(self.device) for v in batch.value], dim=0)
        else:
            values = torch.cat(batch.value, dim=0).to(self.device)
            
        # 处理动作输出
        action_out = list(zip(*batch.action_out))
        action_out = [torch.cat(a, dim=0).to(self.device) for a in action_out]

        # alive_masks: [batch_size * n]
        alive_masks = torch.tensor(
            np.concatenate([item['alive_mask'] for item in batch.misc]), 
            device=self.device
        ).view(-1)

        # 初始化各种回报张量在GPU上
        coop_returns = torch.zeros(batch_size, n, device=self.device) 
        ncoop_returns = torch.zeros(batch_size, n, device=self.device) 
        returns = torch.zeros(batch_size, n, device=self.device)
        advantages = torch.zeros(batch_size, n, device=self.device)
        values = values.view(batch_size, n)

        prev_coop_return = 0
        prev_ncoop_return = 0

        for i in reversed(range(rewards.size(0))):
            coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
            ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]

            prev_coop_return = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                        + ((1 - self.args.mean_ratio) * ncoop_returns[i]) 

        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i] 

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
            
        # element of log_p_a: [(batch_size*n) * num_actions[i]]
        log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
        # actions: [(batch_size*n) * dim_actions]
        actions = actions.contiguous().view(-1, dim_actions)

        if self.args.advantages_per_action:
            # log_prob: [(batch_size*n) * dim_actions]
            log_prob = multinomials_log_densities(actions, log_p_a)
            # the log prob of each action head is multiplied by the advantage
            action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
            action_loss *= alive_masks.unsqueeze(-1)
        else:
            # log_prob: [(batch_size*n) * 1]
            log_prob = multinomials_log_density(actions, log_p_a)
            action_loss = -advantages.view(-1) * log_prob.squeeze()
            action_loss *= alive_masks

        action_loss = action_loss.sum()
        stat['action_loss'] = action_loss.item()

        # value loss term
        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        value_loss *= alive_masks
        value_loss = value_loss.sum()

        stat['value_loss'] = value_loss.item()
        loss = action_loss + self.args.value_coeff * value_loss

        # entropy regularization term
        entropy = 0
        for i in range(len(log_p_a)):
            entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
        stat['entropy'] = entropy.item()
        if self.args.entr > 0:
            loss -= self.args.entr * entropy

        loss.backward()

        return stat

    def run_batch(self, epoch): 
        batch = []
        self.stats = dict()
        self.stats['num_episodes'] = 0
        while len(batch) < self.args.batch_size:
            if self.args.batch_size - len(batch) <= self.args.max_steps:
                self.last_step = True
            episode, episode_stat = self.get_episode(epoch)
            merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1
            batch += episode

        self.last_step = False
        self.stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))
        return batch, self.stats

    def train_batch(self, epoch):
        batch, stat = self.run_batch(epoch)
        self.optimizer.zero_grad()

        s = self.compute_grad(batch)
        merge_stat(s, stat)
        
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= stat['num_steps']
        self.optimizer.step()

        return stat

    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict(),
            'device': self.device
        }

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state['optimizer'])
        # 设备可能有所不同，所以不直接加载