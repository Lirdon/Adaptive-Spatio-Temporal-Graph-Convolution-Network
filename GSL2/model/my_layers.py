import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.nn import GCNConv
import torch.nn.init as init
import torch
from torch import nn
from torch.nn import MultiheadAttention
from GSL.GSL import GraphLearner

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NodeEmbedding(nn.Module):
    def __init__(self, feature_dim, max_len):
        super(NodeEmbedding, self).__init__()
        self.fno = FNO1d(feature_dim, width=max_len)

    def forward(self, x):
        output_list = []
        for i in range(x.shape[-1]):  # 按照最后一维进行循环
            # 取出第i个node的数据，形状为(batchsize, feature, seq_len)
            node_data = x[..., i]
            # 交换后两个维度，改变形状为(batchsize, seq_len, feature)
            node_data = node_data.permute(0, 2, 1)
            # 输入到层中，得到输出，形状为(batchsize, seq_len, out_feature)
            node_output = self.fno(node_data)
            # 输出的后两个维度交换回去，改变形状为(batchsize, out_feature, seq_len)
            node_output = node_output.permute(0, 2, 1)
            # 将处理后的结果添加到列表中
            output_list.append(node_output)

        # 沿着最后一维度重新拼接tensor，形状变为(batchsize, out_feature, seq_len, node_num)
        output_data = torch.stack(output_list, dim=-1)
        return output_data

class FNO1d(nn.Module):
    def __init__(self, feature_dim, width, modes=16):
        super(FNO1d, self).__init__()

        self.modes1 = modes
        self.width = width

        ## 提升维度部分
        self.fc0 = nn.Linear(feature_dim, 128)  # input channel is 2: (a(x), x)
        init.xavier_normal_(self.fc0.weight)
        ## 傅立叶层部分
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        init.xavier_normal_(self.w0.weight)
        init.xavier_normal_(self.w1.weight)
        init.xavier_normal_(self.w2.weight)
        init.xavier_normal_(self.w2.weight)

        ## 降低维度部分
        self.fc1 = nn.Linear(128, 128)
        init.xavier_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(128, 16)
        init.xavier_normal_(self.fc2.weight)

    def forward(self, x):

        x = self.fc0(x)     #[8, 64, 3]->[8,64,64]
        #x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.tanh(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.tanh(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.tanh(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        #x = x.permute(0, 2, 1)

        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        return x


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)  # 傅立叶变换，维度 (20, 64, 65)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device,
                             dtype=torch.cfloat)  # 注意 dtype = torch.cfloat 是复数 维度 (20, 64, 65)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

class TimeEncode(torch.nn.Module):
    def __init__(self, args, factor=5):
        super(TimeEncode, self).__init__()
        #init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])

        time_dim = args.time_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

        #self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)

        #torch.nn.init.xavier_normal_(self.dense.weight)

    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)# [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1) # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)

        return harmonic #self.dense(harmonic)

#在这个层前面把特征投射到高纬度，然后每个维度
class NodeAttention(nn.Module):
    def __init__(self, args):
        super(NodeAttention, self).__init__()
        #self.embed_dim = embed_dim
        self.timeencode = TimeEncode(args, args.enable_cuda)

        '''
        self.W_q = nn.Linear(self.total_dim, self.total_dim)
        self.W_k = nn.Linear(self.total_dim, self.total_dim)
        self.W_v = nn.Linear(self.total_dim, self.total_dim)
        init.xavier_uniform_(self.W_q.weight)
        init.xavier_uniform_(self.W_k.weight)
        init.xavier_uniform_(self.W_v.weight)

        # 如果你也想初始化偏差项，你可以设置它们为0，如下：
        self.W_q.bias.data.fill_(0)
        self.W_k.bias.data.fill_(0)
        self.W_v.bias.data.fill_(0)
        '''
    def forward(self, node_data, node_t):
        # 输入的input_features的形状应为[batch_size, embed_dim, time_step]
        # 输入的t形状应该是[batch_size, time_step]

        # [N, L]

        embed_t = self.timeencode(node_t)        #[N, L, time_dim]可以尝试换成旋转位置嵌入
        embed_t = embed_t.permute(0, 2, 1)         #[N, time_dim, L]
        node_data = node_data + embed_t

        return node_data

#[batch_size, num_heads, tgt_len, src_len]的权重矩阵，通过平均值只保留后面两个维度[tgt_len, src_len]，这个新矩阵的每一个元素代表了在所有批次和所有注意力头下，生成目标序列中某一位置（行索引指示）时，源序列的各位置（列索引指示）的平均贡献程度。用热力图来描述权重矩阵结果
#在测试的时候输出att_weight，另外这个输出的是每个节点的矩阵，需要在节点级别再平均一下


class MyAttentionLayer(nn.Module):
    def __init__(self, args):
        super(MyAttentionLayer, self).__init__()
        #self.node_size = node_size
        self.node_attention = NodeAttention(args)
        self.device = torch.device('cuda' if args.enable_cuda else 'cpu')
        self.total_dim = args.embed_dim + args.time_dim
        self.vertex_num = args.n_vertex
        #self.total_dim = args.embed_dim
    '''
        def forward(self, input_data, t):
        # 输入的input_data的形状应为[batch_size, embed_dim, time_step, node_size]
        # 输入的t的形状应为[batch_size, time_step, node_size]
        output_data = []
        award_mex = []
        node_size = input_data.shape[3]
        print("node_size", node_size)
        indices = [0, 3, 4, 6, 10, 12, 14, 17, 19, 21, 23, 24, 25, 28, 29, 30]
        for i in indices:
            node_data = input_data[:, :, :, i]

            # node_data的形状是[batch_size, embed_dim, time_step]
            node_t = t[:, :, i]
            # node_t的形状是[batch_size, time_step]
            output_node_data, award = self.node_attention(node_data, node_t)

            # output_node_data的形状为[batch_size, time_step, embed_dim]
            output_data.append(output_node_data.unsqueeze(3))
            award_mex.append(award.unsqueeze(1))

        return torch.cat(output_data, dim=3), torch.cat(award_mex, dim=1)
    '''

    def forward(self, input_data, t):
        # 输入的input_data的形状应为[batch_size, embed_dim, time_step, node_size]
        # 输入的t的形状应为[batch_size, time_step, node_size]
        output_data = []

        indices = torch.arange(0, self.vertex_num).to(self.device)

        for i in indices:
            node_data = input_data[:, :, :, i]

            # node_data的形状是[batch_size, embed_dim, time_step]
            node_t = t[:, :, i]
            # node_t的形状是[batch_size, time_step]

            output_node_data = self.node_attention(node_data, node_t)
            # output_node_data = self.node_attention(node_data, node_t, score)
            # output_node_data的形状为[batch_size, time_step, embed_dim]
            output_data.append(output_node_data.unsqueeze(3))
        output_data = torch.cat(output_data, dim=3)
        return output_data

class edge_steam_embed(nn.Module):
    def __init__(self, steam_feature_num):
        super(edge_steam_embed, self).__init__()
        self.fc1 = nn.Linear(steam_feature_num, 8)
        self.fc2 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
    def reset_parameters(self):
        init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.fc1.bias, 0)
        init.kaiming_uniform_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.fc2.bias, 0)

    def forward(self, weight1, weight2):

        weight1 = F.normalize(weight1, p=2, dim=-1)
        weight2 = F.normalize(weight2, p=2, dim=-1)
        weight1 = weight1.unsqueeze(dim=1)
        weight2 = weight2.unsqueeze(dim=1)
        x = torch.cat((weight1, weight2), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = torch.squeeze(x)
        return x

class edge_air_embed(nn.Module):
    def __init__(self, steam_feature_num):
        super(edge_air_embed, self).__init__()
        self.fc1 = nn.Linear(steam_feature_num, 8)
        self.fc2 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
    def reset_parameters(self):
        init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.fc1.bias, 0)
        init.kaiming_uniform_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.fc2.bias, 0)

    def forward(self, weight1, weight2):
        weight1 = F.normalize(weight1, p=2, dim=-1)
        weight2 = F.normalize(weight2, p=2, dim=-1)
        weight1 = weight1.unsqueeze(dim=1)
        weight2 = weight2.unsqueeze(dim=1)
        x = torch.cat((weight1, weight2), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = torch.squeeze(x)
        return x

class edge_ele_embed(nn.Module):
    def __init__(self, steam_feature_num):
        super(edge_ele_embed, self).__init__()
        self.fc1 = nn.Linear(steam_feature_num, 8)
        self.fc2 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
    def reset_parameters(self):
        init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.fc1.bias, 0)
        init.kaiming_uniform_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.fc2.bias, 0)

    def forward(self, weight1, weight2, weight3):
        weight1 = F.normalize(weight1, p=2, dim=-1)
        weight2 = F.normalize(weight2, p=2, dim=-1)
        weight3 = F.normalize(weight3, p=2, dim=-1)
        weight1 = weight1.unsqueeze(dim=1)
        weight2 = weight2.unsqueeze(dim=1)
        weight3 = weight3.unsqueeze(dim=1)
        x = torch.cat((weight1, weight2, weight3), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = torch.squeeze(x)
        return x

#在测试时输出边投射后的值
class GraphConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super(GraphConvLayer, self).__init__()
        self.use_gsl = args.use_gsl
        self.conv1 = GCNConv(in_channels, 128)
        self.dropout1 = nn.Dropout(p=0.1)
        self.conv2 = GCNConv(128, 256)
        self.dropout2 = nn.Dropout(p=0.1)
        self.conv3 = GCNConv(256, out_channels)
        self.relu = nn.ReLU()
        self.graph_skip_conn = args.graph_skip_conn
        self.graph_include_self = args.graph_include_self
        self.num_node_features = in_channels
        self.hidden_dim = args.hidden_dim
        self.graph_metric_type = args.graph_metric_type
        self.graph_type = args.graph_type
        self.top_k = args.top_k
        self.epsilon = args.epsilon
        self.num_per = args.num_per
        self.phy_top_type = args.phy_top_type
        #self.weight = torch.sigmoid(nn.Parameter(torch.tensor(0.0)))
        self.graph_learner = GraphLearner(input_size=self.num_node_features, hidden_size=self.hidden_dim,
                                          graph_type=self.graph_type, top_k=self.top_k,
                                          epsilon=self.epsilon, num_pers=self.num_per,
                                          metric_type=self.graph_metric_type,
                                          feature_denoise=args.feature_denoise, device=device)

    def weighted_sum_of_matrices(self, A, B, w=None):   #搞一个门控算法或者可训练变量来控制
        '''
        #C = torch.zeros_like(A)
        mask = (A != 0) & (B != 0)
        C[mask] = w * A[mask] + (1-w) * B[mask]
        '''
        # w = self.weight
        C = A*w + B*(1-w)
        return C

    def build_epsilon_neighbourhood(self, attention):
        remain_att = attention
        exp_x = torch.exp(attention)
        row_sums = exp_x.sum(dim=1, keepdim=True)
        attention = exp_x / row_sums
        mean = attention.mean(dim=1, keepdim=True)
        std = attention.std(dim=1, keepdim=True)
        multiplier = 1
        epsilon = mean + multiplier * std
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = remain_att * mask + 0 * (1 - mask)
        return weighted_adjacency_matrix

    def build_knn_neighbourhood(self, attention, top_k):
        top_k = min(top_k, attention.size(-1))
        knn_val, knn_ind = torch.topk(attention, top_k, dim=-1)
        weighted_adjacency_matrix = (0 * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val)
        weighted_adjacency_matrix = weighted_adjacency_matrix.to(device)

        return weighted_adjacency_matrix

    def learn_graph(self, node_features, graph_skip_conn=None, graph_include_self=False, init_adj=None):
        new_feature, new_adj = self.graph_learner(node_features)
        if self.phy_top_type == 'epsilon':
            phy_adj = self.build_epsilon_neighbourhood(init_adj)
        elif self.phy_top_type == 'KNN':
            phy_adj = self.build_knn_neighbourhood(init_adj, 1)

        if graph_include_self:
            if torch.cuda.is_available():
                new_adj = new_adj + torch.eye(new_adj.size(0)).cuda()
            else:
                new_adj = new_adj + torch.eye(new_adj.size(0))

        new_adj = self.weighted_sum_of_matrices(phy_adj, new_adj, graph_skip_conn)
        #new_adj = graph_skip_conn * init_adj + (1 - graph_skip_conn) * new_adj

        return new_feature, new_adj

    def forward(self, x, edge_index, edge_weight):
        batch_size, num_features, time_steps, vertex = x.size()
        # Initialize an empty list to store the results for each time step
        outputs = []
        #edge_weight = edge_weight.float()
        # Perform graph convolution for each time step
        for t in range(time_steps):
            batch_list = []
            # Get the input for the current time step
            x_t = x[:, :, t, :].transpose(1, 2)  # [batch_size, vertex, num_features]

            for i in range(x_t.shape[0]):
                if self.use_gsl:
                    raw_adj = to_dense_adj(edge_index, edge_attr=edge_weight)[0]    #写成一个接口，多来几层，每层直接用门口连接

                    x_feature, new_adj = self.learn_graph(node_features=x_t[i],
                                                            graph_skip_conn=self.graph_skip_conn,
                                                            graph_include_self=self.graph_include_self,
                                                            init_adj=raw_adj)
                    edge_index, edge_weight = dense_to_sparse(new_adj)
                else:
                    x_feature = x_t[i]
                x_feature = self.conv1(x_feature, edge_index, edge_weight)
                x_feature = self.dropout1(x_feature)
                x_feature = self.conv2(x_feature, edge_index, edge_weight)
                x_feature = self.dropout2(x_feature)
                x_feature = self.conv3(x_feature, edge_index, edge_weight)
                x_feature = self.relu(x_feature)
                batch_list.append(x_feature)
            batch_x = torch.stack(batch_list, dim=0)
            # Reshape x_t back to [batch_size, num_features, vertex]
            batch_x = batch_x.transpose(1, 2).contiguous()

            # Append the result to the outputs list
            outputs.append(batch_x)

        # Stack the outputs along the time dimension
        outputs = torch.stack(outputs, dim=2)
        return outputs



class AutoRegressiveATT(nn.Module):
    def __init__(self, feature_size, output_size, seq_len, node_num, num_heads):
        super(AutoRegressiveATT, self).__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.node_num = node_num

        # 定义一个多头自注意力层
        self.attention = nn.MultiheadAttention(embed_dim=feature_size*node_num, num_heads=num_heads)
        self.fc = nn.Linear(feature_size, feature_size)  # 线性层输出预测值
        self.activation = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, use_teacher_forcing=True):
        batch_size = x.size(0)
        outputs = []

        # 第一个时间步直接使用输入数据，不通过模型生成
        first_time_step = x[:, :, :1, :]  # 直接使用输入的第一个时间步
        outputs.append(first_time_step)  # 将第一个时间步的输入数据加入输出
        # 创建一个 [seq_len, seq_len] 的掩码，用于自注意力
        mask = torch.tril(torch.ones(self.seq_len, self.seq_len)).to(x.device)
        for t in range(self.seq_len-1):
            if use_teacher_forcing:
                # 使用真实的输入序列
                input_seq = x[:, :, :t + 1, :]  # 使用前 t+1 个时间步的真实数据
            else:
                input_seq = torch.cat(outputs, dim=2)

            input_seq = input_seq.permute(2, 0, 1, 3).reshape(t + 1, batch_size, -1)

            attn_output, _ = self.attention(input_seq, input_seq, input_seq, attn_mask=mask[:t + 1, :t + 1])

            pred = attn_output.permute(1, 2, 0)
            pred = self.avg_pool(pred).reshape(batch_size*self.node_num, self.feature_size)
            pred = self.fc(pred)
            pred = self.activation(pred)  # 应用ReLU激活函数
            pred = pred.reshape(batch_size, self.feature_size, 1, self.node_num)
            outputs.append(pred)
        outputs = torch.cat(outputs, dim=2)

        return outputs

class CVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, seqlen, node):
        super(CVAE, self).__init__()
        self.seqlen = seqlen
        self.node = node

        # Encoder
        self.encoder_fc1 = nn.Linear(input_dim * seqlen * node, hidden_dim)
        self.encoder_fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_fc2_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, input_dim * node)

    def encode(self, x):
        h = F.relu(self.encoder_fc1(x))
        mu = self.encoder_fc2_mu(h)
        logvar = self.encoder_fc2_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x=None, use_teacher_forcing=False):
        batch_size = z.size(0)
        outputs = []
        hidden = F.relu(self.decoder_fc1(z))  # Initial decoder hidden state
        output = self.decoder_fc2(hidden).view(batch_size, 1, -1, self.node)  # First time step output

        outputs.append(output)

        if use_teacher_forcing:
            # Use x directly as target since x == target
            for t in range(1, self.seqlen):
                output = x[:, :, t, :].view(batch_size, 1, -1, self.node)
                outputs.append(output)
        else:
            # Generate recursively using the last generated output
            for t in range(1, self.seqlen):
                hidden = F.relu(self.decoder_fc1(z))
                output = self.decoder_fc2(hidden).view(batch_size, 1, -1, self.node)
                outputs.append(output)

        return torch.cat(outputs, dim=1).permute(0, 2, 1, 3)  # Concatenate outputs along the sequence dimension

    def forward(self, x, use_teacher_forcing=False):
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size, -1)  # Flatten input
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, x, use_teacher_forcing)


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, seqlen, node):
        super(RNNModel, self).__init__()
        self.input_dim = input_dim
        self.seqlen = seqlen
        self.node = node
        self.hidden_dim = hidden_dim
        # RNN takes input_dim * node as the input for each time step
        self.rnn = nn.GRU(input_dim * node, hidden_dim, batch_first=True)

        # Fully connected layer to map hidden state back to input_dim * node
        self.fc = nn.Linear(hidden_dim, input_dim * node)

    def forward(self, x, use_teacher_forcing=False):
        batch_size = x.size(0)

        # Flatten node and input_dim into a single dimension for RNN processing
        x = x.reshape(batch_size, self.seqlen, -1)  # Shape: [batchsize, seqlen, input_dim * node]

        h0 = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)  # Initialize hidden state

        outputs = []
        out, h = self.rnn(x, h0)  # RNN forward pass, shape: [batchsize, seqlen, hidden_dim]
        out = self.fc(out)  # Map hidden state to input_dim * node, shape: [batchsize, seqlen, input_dim * node]

        # Reshape back to the original shape: [batchsize, seqlen, input_dim, node]
        return out.view(batch_size, self.input_dim, self.seqlen, self.node)

class STConvBlock(nn.Module):
    # STConv Block contains 'TGTND' structure
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv or GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    def __init__(self, args, last_block_channel, channels):
        super(STConvBlock, self).__init__()
        if args.time_model == 'attention':
            self.tmp_conv1 = AutoRegressiveATT(last_block_channel, channels[0], args.max_len, args.n_vertex, args.num_heads)
        elif args.time_model == 'CVAE':
            self.tmp_conv1 = CVAE(last_block_channel, 64, 8, args.max_len, args.n_vertex)
        else:
            self.tmp_conv1 = RNNModel(last_block_channel, 64, args.max_len, args.n_vertex)
        self.graph_conv = GraphConvLayer(last_block_channel, channels[1], args)
        if args.time_model == 'attention':
            self.tmp_conv2 = AutoRegressiveATT(channels[1], channels[1], args.max_len, args.n_vertex, args.num_heads)
        elif args.time_model == 'CVAE':
            self.tmp_conv2 = CVAE(channels[1], 64, 8, args.max_len, args.n_vertex)
        else:
            self.tmp_conv2 = RNNModel(channels[1], 64, args.max_len, args.n_vertex)
        self.tc2_ln = nn.LayerNorm([args.n_vertex, channels[2]])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=args.droprate)

    def forward(self, x, edge_index, edge_weight, use_teacher_forcing):

        x = self.tmp_conv1(x, use_teacher_forcing)
        x = self.graph_conv(x, edge_index, edge_weight)
        x = self.tmp_conv2(x, use_teacher_forcing)
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)
        return x


class OutputLayer(nn.Module):
    def __init__(self, feature_size, output_dims):
        super(OutputLayer, self).__init__()
        self.feature_size = feature_size
        self.activation = nn.ReLU()
        # 确保 node_num 可以按块划分

        # 定义 5 个全连接层，每个块对应一个全连接层
        self.fc_layers = nn.ModuleList([nn.Linear(feature_size, out_dim) for out_dim in output_dims])

    def forward(self, x):
        # 假设输入 x 形状为 [batch_size, feature_size, seq_len, node_num]
        # 按照 node_num 维度将 x 分成 5 块
        x_split = torch.split(x, [25, 3, 9, 1, 7, 7, 1], dim=-1)  # 假设 node_num 可以划分为 5 块
        # 对每个块分别应用全连接层
        out_list = []
        for i, fc in enumerate(self.fc_layers):

            #x_chunk = x_split[i].squeeze(-1)  # 形状 [batch_size, feature_size, seq_len]
            x_chunk = x_split[i].permute(0,3,2,1)
            out_chunk = fc(x_chunk)  # 形状 [batch_size, output_dim, seq_len]
            out_chunk = self.activation(out_chunk)
            out_list.append(out_chunk.permute(0,3,2,1))
        # 将输出的不同块拼接在一起

        return out_list