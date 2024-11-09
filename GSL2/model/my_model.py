import torch
import torch.nn as nn
from model import my_layers
from script.utility import selected_pipe


class STGCNGraphConv(nn.Module):
    def __init__(self, args, blocks):
        super(STGCNGraphConv, self).__init__()
        self.device = torch.device('cuda' if args.enable_cuda else 'cpu')
        self.steam_embed = my_layers.NodeEmbedding(args.steam_dim, args.max_len)
        self.dhs_embed = my_layers.NodeEmbedding(args.dhs_dim, args.max_len)
        self.air_embed = my_layers.NodeEmbedding(args.air_dim, args.max_len)
        self.ele_embed = my_layers.NodeEmbedding(args.ele_dim, args.max_len)
        self.c1_embed = my_layers.NodeEmbedding(args.c1_dim, args.max_len)
        self.c2_embed = my_layers.NodeEmbedding(args.c2_dim, args.max_len)
        self.battery_embed = my_layers.NodeEmbedding(args.battery_dim, args.max_len)

        self.steam_edge_embed = my_layers.edge_steam_embed(2)
        self.air_edge_embed = my_layers.edge_air_embed(2)
        self.ele_edge_embed = my_layers.edge_ele_embed(3)

        self.attention = my_layers.MyAttentionLayer(args)
        self.stblock1 = my_layers.STConvBlock(args, blocks[0][-1], blocks[0 + 1])
        self.stblock2 = my_layers.STConvBlock(args, blocks[1][-1], blocks[1 + 1])

        self.bn0 = nn.BatchNorm2d(blocks[0][-1])
        self.bn1 = nn.BatchNorm2d(blocks[1][-1])  # 根据输出通道数设置
        self.bn2 = nn.BatchNorm2d(blocks[2][-1])  # 根据输出通道数设置

        steam1, steam2, dhs1, dhs2, air1, air2, ele1, ele2, ele3, edge_index = selected_pipe()
        self.edge_index = torch.tensor(edge_index).to(self.device)
        self.steam1 = torch.tensor(steam1, dtype=torch.float32, requires_grad=args.need_grad).to(self.device)
        self.steam2 = torch.tensor(steam2, dtype=torch.float32, requires_grad=args.need_grad).to(self.device)
        self.dhs1 = torch.tensor(dhs1, dtype=torch.float32, requires_grad=args.need_grad).to(self.device)
        self.dhs2 = torch.tensor(dhs2, dtype=torch.float32, requires_grad=args.need_grad).to(self.device)
        self.air1 = torch.tensor(air1, dtype=torch.float32, requires_grad=args.need_grad).to(self.device)
        self.air2 = torch.tensor(air2, dtype=torch.float32, requires_grad=args.need_grad).to(self.device)
        self.ele1 = torch.tensor(ele1, dtype=torch.float32, requires_grad=args.need_grad).to(self.device)
        self.ele2 = torch.tensor(ele2, dtype=torch.float32, requires_grad=args.need_grad).to(self.device)
        self.ele3 = torch.tensor(ele3, dtype=torch.float32, requires_grad=args.need_grad).to(self.device)
        self.conc = nn.Parameter(
            torch.nn.functional.normalize(
                torch.randn(9, dtype=torch.float32, requires_grad=args.need_grad).to(self.device),
                p=2, dim=0
            )
        )
        t = [10 * i for i in range(args.max_len)]
        t = torch.tensor(t, dtype=torch.float32, requires_grad=args.need_grad).to(self.device)
        t = torch.unsqueeze(t, dim=-1)
        self.t = t.expand(args.max_len, args.n_vertex)

        self.output_layer = my_layers.OutputLayer(args.graph_dim, [args.steam_dim, args.c1_dim, args.ele_dim, args.c2_dim, args.air_dim, args.dhs_dim, args.battery_dim])

    def forward(self, xs, xa, xe, xc1, xc2, xd, xb, use_teacher_forcing):

        steam_weight = self.steam_edge_embed(self.steam1, self.steam2)
        dhs_weight = self.steam_edge_embed(self.dhs1, self.dhs2)
        air_weight = self.air_edge_embed(self.air1, self.air2)
        ele_weight = self.ele_edge_embed(self.ele1, self.ele2, self.ele3)
        edge_weight = torch.cat([steam_weight, ele_weight, air_weight, self.conc, dhs_weight], dim=0)#这里还要补充中间边和正确维度
   
        xs = self.steam_embed(xs)
        
        xd = self.dhs_embed(xd)
        xa = self.air_embed(xa)
        xe = self.ele_embed(xe)
        xc1 = self.c1_embed(xc1)
        xc2 = self.c2_embed(xc2)
        xb = self.battery_embed(xb)


        x = torch.cat((xs, xc1, xe, xc2, xa, xd, xb), dim=3)
        
        t_expanded = self.t.unsqueeze(0)
        t = t_expanded.repeat_interleave(xs.shape[0], dim=0)

        x = self.attention(x, t)
        
        x = self.stblock1(x, self.edge_index, edge_weight, use_teacher_forcing)
        x = self.bn1(x)
        x = self.stblock2(x, self.edge_index, edge_weight, use_teacher_forcing)
        x = self.bn2(x)
        x = self.output_layer(x)
        return x







