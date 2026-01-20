from torch import nn
import torch
import ipdb
import math
import torch.nn.functional as F
#Adapter_module_GRN
# 调整后的GRN层（兼容2D/3D特征）
class GlobalResponseNorm(nn.Module):
    """
    全局响应归一化层（兼容2D特征：(B, C) 和 3D特征：(B, L, C)）
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # 处理不同维度的输入：2D (B, C) 或 3D (B, L, C)
        if x.dim() == 2:
            # 2D特征：无空间维度，直接计算通道维度的均值（简化处理）
            x_g = x.norm(p=2, dim=-1, keepdim=True)
            x_n = x_g / (x_g.mean(dim=-1, keepdim=True) + self.eps)
        elif x.dim() == 3:
            # print("xinput.shape",x.shape)
            # 3D特征：将序列长度L视为“空间维度”，计算L维度的L2范数
            spatial_dim = 1  # 对应序列长度L的维度（B, L, C）
            channel_dim = -1 # 对应通道C的维度
            x_g = x.norm(p=2, dim=spatial_dim, keepdim=True)
            x_n = x_g / (x_g.mean(dim=channel_dim, keepdim=True) + self.eps)
        else:
            raise ValueError(f"GRN仅支持2D/3D输入，当前输入维度：{x.dim()}")
        
        # 广播权重和偏置（适配输入形状）
        weight = self.weight.view(1, 1, -1) if x.dim() == 3 else self.weight.view(1, -1)
        bias = self.bias.view(1, 1, -1) if x.dim() == 3 else self.bias.view(1, -1)
        
        # 加权融合
        out = x + torch.addcmul(bias, weight, x * x_n)
        return out
    
class SimpleAdapter(nn.Module):
    def __init__(self, c_in, c_out=768):
        super(SimpleAdapter, self).__init__()
        self.fc = nn.Linear(c_in, c_out, bias=False)
        self.grn = GlobalResponseNorm(dim=c_out)  # 全连接后插入GRN
        self.act = nn.LeakyReLU()

    def forward(self, x):
        # 输入形状：2D (B, c_in) 或 3D (B, L, c_in)
        x = self.fc(x)       # 全连接层
        x = self.grn(x)      # GRN归一化（核心位置）
        x = self.act(x)      # 激活函数
        return x

class SimpleProj(nn.Module):
    def __init__(self, c_in, c_out=768, relu=True):
        super(SimpleProj, self).__init__()
        self.fc = nn.Linear(c_in, c_out, bias=False)
        self.grn = GlobalResponseNorm(dim=c_out)  # 全连接后插入GRN
        self.relu = relu
        if relu:
            self.act = nn.LeakyReLU()

    def forward(self, x):
        # 输入形状：2D (B, c_in) 或 3D (B, L, c_in)
        x = self.fc(x)       # 全连接层
        x = self.grn(x)      # GRN归一化（核心位置）
        if self.relu:
            x = self.act(x)  # 激活函数（若有）
        return x