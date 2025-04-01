import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from action_utils import select_action, translate_action
from gnn_layers import GraphAttention

def apply_delta_transformation(probs, delta=1.0):
    """
    应用 delta 变换到概率分布
    probs: softmax 后的概率分布
    delta: 超参数，控制变换强度
    """
    # 计算分母：Σ_i a_i^δ
    denominator = torch.sum(probs ** delta, dim=-1, keepdim=True)
    # 应用 delta 变换公式: a_i^δ / Σ_i a_i^δ
    transformed_probs = (probs ** delta) / denominator
    return transformed_probs

class MAGIC(nn.Module):
    """
    The communication protocol of Multi-Agent Graph AttentIon Communication (MAGIC)
    """
    def __init__(self, args):
        super(MAGIC, self).__init__()
        """
        Initialization method for the MAGIC communication protocol (2 rounds of communication)

        Arguements:
            args (Namespace): Parse arguments
        """

        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        
        # 设置设备
        self.device = torch.device("cpu")
        
        #self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        
        dropout = 0
        negative_slope = 0.2

        # initialize sub-processors
        self.sub_processor1 = GraphAttention(args.hid_size, args.gat_hid_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.gat_num_heads, self_loop_type=args.self_loop_type1, average=False, normalize=args.first_gat_normalize)
        self.sub_processor2 = GraphAttention(args.gat_hid_size*args.gat_num_heads, args.hid_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.gat_num_heads_out, self_loop_type=args.self_loop_type2, average=True, normalize=args.second_gat_normalize)
        
        # initialize the gat encoder for the Scheduler
        if args.use_gat_encoder:
            self.gat_encoder = GraphAttention(args.hid_size, args.gat_encoder_out_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.ge_num_heads, self_loop_type=1, average=True, normalize=args.gat_encoder_normalize)

        # 将原始观察转换为隐藏状态大小的线性层
        self.obs_encoder = nn.Linear(args.obs_size, args.hid_size)

        self.init_hidden(args.batch_size)  # 初始化隐藏状态
        self.lstm_cell = nn.LSTMCell(args.hid_size, args.hid_size)  # 处理时序信息的LSTM单元

        # initialize mlp layers for the sub-schedulers
        if not args.first_graph_complete:
            # 第一轮通信的sub-scheduler1
            if args.use_gat_encoder:
                self.sub_scheduler_mlp1 = nn.Sequential(
                    nn.Linear(args.gat_encoder_out_size*2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, 2))
            else:
                self.sub_scheduler_mlp1 = nn.Sequential(
                    nn.Linear(self.hid_size*2, self.hid_size//2),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//2, self.hid_size//8),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//8, 2))
                
        if args.learn_second_graph and not args.second_graph_complete:
            # 第二轮通信的sub-scheduler2
            if args.use_gat_encoder:
                self.sub_scheduler_mlp2 = nn.Sequential(
                    nn.Linear(args.gat_encoder_out_size*2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, 2))
            else:
                self.sub_scheduler_mlp2 = nn.Sequential(
                    nn.Linear(self.hid_size*2, self.hid_size//2),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//2, self.hid_size//8),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//8, 2))

        if args.message_encoder:
            self.message_encoder = nn.Linear(args.hid_size, args.hid_size)
        if args.message_decoder:
            self.message_decoder = nn.Linear(args.hid_size, args.hid_size)

        # initialize weights as 0  # 初始化相关模块的权重为0
        if args.comm_init == 'zeros':
            if args.message_encoder:
                self.message_encoder.weight.data.zero_()
            if args.message_decoder:
                self.message_decoder.weight.data.zero_()
            if not args.first_graph_complete:
                self.sub_scheduler_mlp1.apply(self.init_linear)
            if args.learn_second_graph and not args.second_graph_complete:
                self.sub_scheduler_mlp2.apply(self.init_linear)
                   
        # initialize the action head (in practice, one action head is used)
        self.action_heads = nn.ModuleList([nn.Linear(2*args.hid_size, o)
                                        for o in args.naction_heads])
        # initialize the value head
        self.value_head = nn.Linear(2 * self.hid_size, 1)
        
        # 将所有模块移到指定设备
        self.to(self.device)


    def forward(self, x, info={}):
        """
        Forward function of MAGIC (two rounds of communication)

        Arguments:
            x (list): a list for the input of the communication protocol [observations, (previous hidden states, previous cell states)]
            observations (tensor): the observations for all agents [1 (batch_size) * n * obs_size]
            previous hidden/cell states (tensor): the hidden/cell states from the previous time steps [n * hid_size]

        Returns:
            action_out (list): a list of tensors of size [1 (batch_size) * n * num_actions] that represent output policy distributions
            value_head (tensor): estimated values [n * 1]
            next hidden/cell states (tensor): next hidden/cell states [n * hid_size]
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

        value_head = self.value_head(torch.cat((hidden_state, comm), dim=-1))

        h = hidden_state.view(batch_size, n, self.hid_size)
        c = comm.view(batch_size, n, self.hid_size)

        action_out = [F.log_softmax(action_head(torch.cat((h, c), dim=-1)), dim=-1) for action_head in self.action_heads]
        
        return action_out, value_head, (hidden_state.clone(), cell_state.clone())

    
    def get_agent_mask(self, batch_size, info):
        """
        Function to generate agent mask to mask out inactive agents (only effective in Traffic Junction)

        Returns:
            num_agents_alive (int): number of active agents
            agent_mask (tensor): [n, 1]
        """

        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n, device=self.device)
            num_agents_alive = n

        agent_mask = agent_mask.view(n, 1).clone()

        return num_agents_alive, agent_mask

    def init_linear(self, m):
        """
        Function to initialize the parameters in nn.Linear as 0
        """
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.)
            m.bias.data.fill_(0.)
        
    def init_hidden(self, batch_size):
        """
        Function to initialize the hidden states and cell states
        """
        return (torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True, device=self.device),
                torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True, device=self.device))
    
    
    def sub_scheduler(self, sub_scheduler_mlp, hidden_state, agent_mask, directed=True):
        """
        Function to perform a sub-scheduler

        Arguments: 
            sub_scheduler_mlp (nn.Sequential): the MLP layers in a sub-scheduler
            hidden_state (tensor): the encoded messages input to the sub-scheduler [n * hid_size]
            agent_mask (tensor): [n * 1]
            directed (bool): decide if generate directed graphs

        Return:
            adj (tensor): a adjacency matrix which is the communication graph [n * n]  
        """

        # hidden_state: [n * hid_size]
        n = self.args.nagents
        hid_size = hidden_state.size(-1)
        
        # 确保 hidden_state 和 agent_mask 在正确设备上
        hidden_state = hidden_state.to(self.device)
        agent_mask = agent_mask.to(self.device)
        
        # hard_attn_input: [n * n * (2*hid_size)]
        hard_attn_input = torch.cat([hidden_state.repeat(1, n).view(n * n, -1), hidden_state.repeat(n, 1)], dim=1).view(n, -1, 2 * hid_size)
        
        # hard_attn_output: [n * n * 2]
        if directed:
            hard_attn_output = F.gumbel_softmax(sub_scheduler_mlp(hard_attn_input), hard=True)
        else:
            hard_attn_output = F.gumbel_softmax(0.5*sub_scheduler_mlp(hard_attn_input)+0.5*sub_scheduler_mlp(hard_attn_input.permute(1,0,2)), hard=True)
            
        # hard_attn_output: [n * n * 1]
        hard_attn_output = torch.narrow(hard_attn_output, 2, 1, 1)
        
        # agent_mask and agent_mask_transpose: [n * n]
        agent_mask = agent_mask.expand(n, n)
        agent_mask_transpose = agent_mask.transpose(0, 1)
        
        # adj: [n * n]
        adj = hard_attn_output.squeeze() * agent_mask * agent_mask_transpose
        
        return adj
    
    def get_complete_graph(self, agent_mask):
        """
        Function to generate a complete graph, and mask it with agent_mask
        """
        n = self.args.nagents
        adj = torch.ones(n, n, device=self.device)
        agent_mask = agent_mask.expand(n, n)
        agent_mask_transpose = agent_mask.transpose(0, 1)
        adj = adj * agent_mask * agent_mask_transpose
        
        return adj
        
    def to(self, device):
        """重写 to 方法确保所有组件都移到正确设备"""
        self.device = device
        return super(MAGIC, self).to(device)