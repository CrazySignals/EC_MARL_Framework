import torch
import numpy as np
from scipy.spatial.distance import cosine, pdist, squareform
import matplotlib.pyplot as plt
from collections import defaultdict
import torch.nn.functional as F
from magic_bad import MAGIC

class MAGICMonitor:
    """
    Monitor for tracking entropy and agent similarity in MAGIC networks
    """
    def __init__(self, nagents, save_dir='./monitor_results/'):
        """
        Initialize the monitoring system
        
        Arguments:
            nagents (int): Number of agents in the network
            save_dir (str): Directory to save monitoring results
        """
        self.nagents = nagents
        self.save_dir = save_dir
        
        # Create storage for metrics
        self.metrics = {
            'hidden_entropy': defaultdict(list),  # Entropy of hidden states by epoch
            'comm_entropy': defaultdict(list),    # Entropy of communication vectors by epoch
            'hidden_similarity': defaultdict(list),  # Inter-agent similarity of hidden states by epoch
            'comm_similarity': defaultdict(list),    # Inter-agent similarity of communication vectors by epoch
            'hidden_stats': defaultdict(dict),    # Basic statistics of hidden states
            'comm_stats': defaultdict(dict)       # Basic statistics of communication vectors
        }
        
        # Ensure save directory exists
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    def calculate_entropy(self, tensor, axis=1):
        """
        Calculate the entropy of a tensor along a specific axis
        
        Arguments:
            tensor (torch.Tensor): Input tensor
            axis (int): Axis along which to calculate entropy
            
        Returns:
            torch.Tensor: Entropy values
        """
        # Convert to probabilities using softmax if values aren't already in [0,1]
        if torch.min(tensor) < 0 or torch.max(tensor) > 1:
            # Normalize the values to get a probability distribution
            probs = torch.softmax(tensor, dim=axis)
        else:
            # Already in [0,1], just normalize to sum to 1
            probs = tensor / torch.sum(tensor, dim=axis, keepdim=True)
        
        # Avoid log(0) by adding a small epsilon
        eps = 1e-10
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=axis)
        return entropy
    
    def calculate_similarity_matrix(self, tensor):
        """
        Calculate pairwise cosine similarity between agent representations
        
        Arguments:
            tensor (torch.Tensor): [batch_size, n_agents, feature_dim] tensor
            
        Returns:
            torch.Tensor: [n_agents, n_agents] similarity matrix
        """
        # Average across batch dimension if needed
        if tensor.dim() == 3:
            tensor = tensor.mean(dim=0)  # [n_agents, feature_dim]
        
        similarity_matrix = torch.zeros((self.nagents, self.nagents))
        for i in range(self.nagents):
            for j in range(i+1, self.nagents):
                # Calculate cosine similarity
                sim = 1 - torch.nn.functional.cosine_similarity(
                    tensor[i].unsqueeze(0), tensor[j].unsqueeze(0), dim=1
                ).item()
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
                
        return similarity_matrix
    
    def record_epoch_metrics(self, epoch, hidden_states, comm_vectors):
        """
        Record metrics for a given epoch
        
        Arguments:
            epoch (int): Current epoch
            hidden_states (torch.Tensor): [batch_size, n_agents, hidden_dim] tensor of hidden states
            comm_vectors (torch.Tensor): [batch_size, n_agents, comm_dim] tensor of communication vectors
        """
        # Ensure tensors are on CPU for numpy conversion
        hidden_states = hidden_states.detach().cpu()
        comm_vectors = comm_vectors.detach().cpu()
        
        # Calculate entropy for each agent's hidden state and comm vector
        h_entropy = self.calculate_entropy(hidden_states)  # [batch_size, n_agents]
        c_entropy = self.calculate_entropy(comm_vectors)   # [batch_size, n_agents]
        
        # Calculate average entropy across batch for each agent
        h_entropy_mean = h_entropy.mean(dim=0)  # [n_agents]
        c_entropy_mean = c_entropy.mean(dim=0)  # [n_agents]
        
        # Record entropy metrics
        self.metrics['hidden_entropy'][epoch].append(h_entropy_mean.numpy())
        self.metrics['comm_entropy'][epoch].append(c_entropy_mean.numpy())
        
        # Calculate similarity matrices
        h_sim = self.calculate_similarity_matrix(hidden_states)
        c_sim = self.calculate_similarity_matrix(comm_vectors)
        
        # Record similarity metrics
        self.metrics['hidden_similarity'][epoch].append(h_sim.numpy())
        self.metrics['comm_similarity'][epoch].append(c_sim.numpy())
        
        # Record basic statistics
        self.metrics['hidden_stats'][epoch] = {
            'mean': hidden_states.mean().item(),
            'std': hidden_states.std().item(),
            'min': hidden_states.min().item(),
            'max': hidden_states.max().item()
        }
        
        self.metrics['comm_stats'][epoch] = {
            'mean': comm_vectors.mean().item(),
            'std': comm_vectors.std().item(),
            'min': comm_vectors.min().item(),
            'max': comm_vectors.max().item()
        }
    
    def save_epoch_metrics(self, epoch):
        """
        Save metrics for the specified epoch to disk
        
        Arguments:
            epoch (int): Epoch to save metrics for
        """
        import pickle
        
        # Save metrics for this epoch
        filename = f"{self.save_dir}/metrics_epoch_{epoch}.pkl"
        
        epoch_metrics = {key: val[epoch] for key, val in self.metrics.items() if epoch in val}
        with open(filename, 'wb') as f:
            pickle.dump(epoch_metrics, f)
    
    def plot_entropy_trends(self, save=True):
        """Plot how entropy changes over epochs"""
        epochs = sorted(self.metrics['hidden_entropy'].keys())
        
        # Process data for plotting
        h_entropies = [np.mean(self.metrics['hidden_entropy'][e]) for e in epochs]
        c_entropies = [np.mean(self.metrics['comm_entropy'][e]) for e in epochs]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, h_entropies, 'b-', label='Hidden State Entropy')
        plt.plot(epochs, c_entropies, 'r-', label='Communication Vector Entropy')
        plt.xlabel('Epoch')
        plt.ylabel('Average Entropy')
        plt.title('Entropy Evolution During Training')
        plt.legend()
        plt.grid(True)
        
        if save:
            plt.savefig(f"{self.save_dir}/entropy_trends.png", dpi=300)
            plt.close()
        else:
            plt.show()
    
    def plot_similarity_heatmaps(self, epoch, save=True):
        """Plot heatmaps of agent similarities for a specified epoch"""
        if epoch not in self.metrics['hidden_similarity']:
            print(f"No data available for epoch {epoch}")
            return
        
        h_sim = np.mean(self.metrics['hidden_similarity'][epoch], axis=0)
        c_sim = np.mean(self.metrics['comm_similarity'][epoch], axis=0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot hidden state similarity
        im1 = ax1.imshow(h_sim, cmap='viridis')
        ax1.set_title(f'Hidden State Similarity (Epoch {epoch})')
        ax1.set_xlabel('Agent Index')
        ax1.set_ylabel('Agent Index')
        plt.colorbar(im1, ax=ax1, label='Distance')
        
        # Plot communication vector similarity
        im2 = ax2.imshow(c_sim, cmap='viridis')
        ax2.set_title(f'Communication Vector Similarity (Epoch {epoch})')
        ax2.set_xlabel('Agent Index')
        ax2.set_ylabel('Agent Index')
        plt.colorbar(im2, ax=ax2, label='Distance')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.save_dir}/similarity_heatmap_epoch_{epoch}.png", dpi=300)
            plt.close()
        else:
            plt.show()
    
    def plot_all_metrics(self):
        """Generate all plots for available epochs"""
        # Plot entropy trends
        self.plot_entropy_trends()
        
        # Plot similarity heatmaps for each epoch
        for epoch in self.metrics['hidden_similarity'].keys():
            self.plot_similarity_heatmaps(epoch)


# Example integration with MAGIC network
def integrate_with_magic(magic_network, monitor):
    """
    Modify the MAGIC network's forward method to record metrics
    
    This function returns a new forward method that wraps the original
    one and records metrics before returning results.
    
    Arguments:
        magic_network (MAGIC): The MAGIC network instance
        monitor (MAGICMonitor): The monitor instance
        
    Returns:
        function: Modified forward method
    """
    # Store the original forward method
    original_forward = magic_network.forward
    
    # Define the wrapper function
    def forward_with_monitoring(x, info={}, epoch=None):
        # Call the original forward method
        action_out, value_head, (hidden_state, cell_state) = original_forward(x, info)
        
        if epoch is not None:
            # Reshape hidden_state and comm to [batch_size, n_agents, dim] if needed
            batch_size = x[0].size(0)
            n = magic_network.nagents
            
            # Get the final hidden state
            h = hidden_state.view(batch_size, n, magic_network.hid_size)
            
            # Get the final communication vector - this is from inside the forward method
            if hasattr(magic_network, 'comm_final'):
                c = magic_network.comm_final.view(batch_size, n, magic_network.hid_size)
            else:
                # We need to recompute the final comm vector if it's not stored
                # This is a simplified version and may need to be updated based on your MAGIC implementation
                comm = hidden_state.clone()
                if magic_network.args.message_encoder:
                    comm = magic_network.message_encoder(comm)
                
                # Apply sub-processor 1 and 2 as in the original forward method
                # This is simplified and may need adjustment
                agent_mask = info.get('agent_mask', torch.ones(n, 1, device=magic_network.device))
                adj1 = magic_network.get_complete_graph(agent_mask)
                comm = torch.nn.functional.elu(magic_network.sub_processor1(comm, adj1))
                
                adj2 = magic_network.get_complete_graph(agent_mask)
                comm = magic_network.sub_processor2(comm, adj2)
                
                if magic_network.args.message_decoder:
                    comm = magic_network.message_decoder(comm)
                
                c = comm.view(batch_size, n, magic_network.hid_size)
            
            # Record metrics
            monitor.record_epoch_metrics(epoch, h, c)
        
        return action_out, value_head, (hidden_state, cell_state)
    
    return forward_with_monitoring


# Modified MAGIC class to expose internal communication state
class ModifiedMAGIC(MAGIC):
    """
    Modified version of MAGIC that exposes communication vectors for monitoring
    """
    def forward(self, x, info={}):
        """
        Modified forward method to store the final communication vector
        """
        # n: number of agents
        obs, extras = x

        # 将输入移到正确设备
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.device)
        elif obs.device != self.device:
            obs = obs.to(self.device)

        # encoded_obs: [1 (batch_size) * n * hid_size]
        encoded_obs = self.obs_encoder(obs)
        
        # 确保额外状态在正确设备上
        if isinstance(extras, tuple):
            hidden_state, cell_state = extras
            if isinstance(hidden_state, np.ndarray):
                hidden_state = torch.from_numpy(hidden_state).to(self.device)
            elif hidden_state.device != self.device:
                hidden_state = hidden_state.to(self.device)
                
            if isinstance(cell_state, np.ndarray):
                cell_state = torch.from_numpy(cell_state).to(self.device)
            elif cell_state.device != self.device:
                cell_state = cell_state.to(self.device)
        else:
            # 如果没有提供状态，初始化
            batch_size = encoded_obs.size(0)
            hidden_state, cell_state = self.init_hidden(batch_size)
            hidden_state = hidden_state.to(self.device)
            cell_state = cell_state.to(self.device)

        batch_size = encoded_obs.size()[0]
        n = self.nagents
        
        # 重要修复：确保隐藏状态维度正确
        if hidden_state.dim() == 3:  # 如果形状是[batch_size, nagents, hid_size]
            hidden_state = hidden_state.view(batch_size * n, self.hid_size)
            cell_state = cell_state.view(batch_size * n, self.hid_size)
        
        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)
        
        # 确保 agent_mask 在正确设备上
        agent_mask = agent_mask.to(self.device)

        # if self.args.comm_mask_zero == True, block the communiction (can also comment out the protocol to make training faster)
        if self.args.comm_mask_zero:
            agent_mask *= torch.zeros(n, 1, device=self.device)

        # LSTM处理
        # reshape encoded_obs以匹配LSTM的输入期望
        encoded_flat = encoded_obs.view(batch_size * n, self.hid_size)
        hidden_state, cell_state = self.lstm_cell(encoded_flat, (hidden_state, cell_state))
        
        # comm: [n * hid_size]
        comm = hidden_state # 隐藏层作为通信信息
        if self.args.message_encoder:
            comm = self.message_encoder(comm) # 如果信息需要二次编码，就将隐藏层放入信息编码器
            
        # mask communcation from dead agents (only effective in Traffic Junction)
        comm = comm * agent_mask
        comm_ori = comm.clone() # 保持原始通信状态供后续使用

        # sub-scheduler 1
        # if args.first_graph_complete == True, sub-scheduler 1 will be disabled
        if not self.args.first_graph_complete: # 构建第一轮通信的图
            if self.args.use_gat_encoder:
                adj_complete = self.get_complete_graph(agent_mask)
                encoded_state1 = self.gat_encoder(comm, adj_complete)
                adj1 = self.sub_scheduler(self.sub_scheduler_mlp1, encoded_state1, agent_mask, self.args.directed)
            else:
                adj1 = self.sub_scheduler(self.sub_scheduler_mlp1, comm, agent_mask, self.args.directed)
        else:
            adj1 = self.get_complete_graph(agent_mask)

        # sub-processor 1
        comm = F.elu(self.sub_processor1(comm, adj1))
        
        # sub-scheduler 2
        if self.args.learn_second_graph and not self.args.second_graph_complete:
            if self.args.use_gat_encoder:
                if self.args.first_graph_complete:
                    adj_complete = self.get_complete_graph(agent_mask)
                    encoded_state2 = self.gat_encoder(comm_ori, adj_complete)
                else:
                    encoded_state2 = encoded_state1
                adj2 = self.sub_scheduler(self.sub_scheduler_mlp2, encoded_state2, agent_mask, self.args.directed)
            else:
                adj2 = self.sub_scheduler(self.sub_scheduler_mlp2, comm_ori, agent_mask, self.args.directed)
        elif not self.args.learn_second_graph and not self.args.second_graph_complete:
            adj2 = adj1
        else:
            adj2 = self.get_complete_graph(agent_mask)
            
        # sub-processor 2
        comm = self.sub_processor2(comm, adj2)
        
        # mask communication to dead agents (only effective in Traffic Junction)
        comm = comm * agent_mask
        
        if self.args.message_decoder:
            comm = self.message_decoder(comm)
            
        # Store the final communication vector for monitoring
        self.comm_final = comm

        value_head = self.value_head(torch.cat((hidden_state, comm), dim=-1))

        h = hidden_state.view(batch_size, n, self.hid_size)
        c = comm.view(batch_size, n, self.hid_size)

        action_out = [F.log_softmax(action_head(torch.cat((h, c), dim=-1)), dim=-1) for action_head in self.action_heads]
        
        return action_out, value_head, (hidden_state.clone(), cell_state.clone())


# # Example usage in training loop
# def example_training_loop_integration():
#     """Example of how to integrate the monitoring into a training loop"""
#     import argparse
    
#     # Setup args (simplified)
#     args = argparse.Namespace()
#     args.nagents = 5
#     args.hid_size = 64
#     args.batch_size = 32
#     args.cuda = True
#     # ... other args as needed
    
#     # Initialize the modified MAGIC network
#     magic_net = ModifiedMAGIC(args)
    
#     # Initialize the monitor
#     monitor = MAGICMonitor(nagents=args.nagents, save_dir='./monitor_results/')
    
#     # Training loop example (simplified)
#     num_epochs = 100
#     for epoch in range(num_epochs):
#         # ... training code ...
        
#         # During evaluation phase, record metrics
#         with torch.no_grad():
#             for batch in eval_dataloader:  # Assume we have an evaluation dataloader
#                 observations, info = batch
#                 extras = magic_net.init_hidden(args.batch_size)
                
#                 # Forward pass with monitoring
#                 _, _, _ = magic_net.forward((observations, extras), info)
                
#                 # Record metrics from the stored comm_final
#                 h = magic_net.hidden_state.view(args.batch_size, args.nagents, args.hid_size)
#                 c = magic_net.comm_final.view(args.batch_size, args.nagents, args.hid_size)
#                 monitor.record_epoch_metrics(epoch, h, c)
        
#         # Save metrics for this epoch
#         monitor.save_epoch_metrics(epoch)
        
#         # Generate plots every 10 epochs
#         if epoch % 10 == 0 or epoch == num_epochs - 1:
#             monitor.plot_entropy_trends()
#             monitor.plot_similarity_heatmaps(epoch)
    
#     # Generate all plots at the end
#     monitor.plot_all_metrics()
class MonitoredMAGIC(ModifiedMAGIC):
    """具有内置监控功能的MAGIC网络，支持多进程环境"""
    
    def __init__(self, args, monitor=None, save_dir='./monitor_results/'):
        """初始化带监控功能的MAGIC网络"""
        super().__init__(args)
        
        # 如果没有提供monitor实例，创建一个新的
        if monitor is None:
            self.monitor = MAGICMonitor(nagents=args.nagents, save_dir=save_dir)
        else:
            self.monitor = monitor
        
        # 确保有记录当前epoch的属性
        self.current_epoch = None
        # 确保comm_final属性存在
        if not hasattr(self, 'comm_final'):
            self.comm_final = None
    
    def set_epoch(self, epoch):
        """设置当前epoch"""
        self.current_epoch = epoch
    
    def forward(self, x, info={}):
        """增强的forward方法，自动收集监控数据"""
        # 调用父类的forward方法
        action_out, value_head, (hidden_state, cell_state) = super().forward(x, info)
        
        # 只有在设置了当前epoch的情况下才收集数据
        if hasattr(self, 'monitor') and self.monitor is not None and self.current_epoch is not None:
            try:
                # 重塑hidden_state和comm为[batch_size, n_agents, dim]
                batch_size = x[0].size(0)
                n = self.nagents
                
                # 获取最终隐藏状态
                h = hidden_state.view(batch_size, n, self.hid_size)
                
                # 获取最终通信向量
                if hasattr(self, 'comm_final') and self.comm_final is not None:
                    c = self.comm_final.view(batch_size, n, self.hid_size)
                    
                    # 记录指标
                    self.monitor.record_epoch_metrics(self.current_epoch, h, c)
            except Exception as e:
                print(f"监控数据收集失败: {e}")
        
        return action_out, value_head, (hidden_state, cell_state)
    
    def save_metrics(self, epoch=None):
        """保存指定epoch的监控指标"""
        if hasattr(self, 'monitor') and self.monitor is not None:
            save_epoch = epoch if epoch is not None else self.current_epoch
            if save_epoch is not None:
                self.monitor.save_epoch_metrics(save_epoch)
    
    def plot_metrics(self, epoch=None, all_metrics=False):
        """生成并保存监控指标的可视化图表"""
        if hasattr(self, 'monitor') and self.monitor is not None:
            if all_metrics:
                self.monitor.plot_all_metrics()
            else:
                self.monitor.plot_entropy_trends()
                plot_epoch = epoch if epoch is not None else self.current_epoch
                if plot_epoch is not None:
                    self.monitor.plot_similarity_heatmaps(plot_epoch)
                    
    def __getstate__(self):
        """自定义序列化方法，确保可以通过多进程传递"""
        state = self.__dict__.copy()
        # 移除不可序列化的对象
        return state
        
    def __setstate__(self, state):
        """自定义反序列化方法"""
        self.__dict__.update(state)