import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange
from src.PreprocessingModule import PreprocessingModule
# import numpy as np
import math

# Residual Connection
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# Layer Normalization
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Attention Mechanism
class Attention(nn.Module):
    def __init__(self, n, num_heads):
        super(Attention, self).__init__()
        self.n = n
        self.num_heads = num_heads
        self.head_dim = n // num_heads
        assert self.head_dim * num_heads == n, "n must be divisible by num_heads"

        self.query = nn.Linear(n, n)
        self.key = nn.Linear(n, n)
        self.value = nn.Linear(n, n)
        self.fc_out = nn.Linear(n, n)

    def forward(self, x):
        batch_size = x.shape[0]

        # Transform input for multi-head attention
        Q = self.query(x).view(batch_size, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, self.num_heads, self.head_dim)

        # Calculate attention scores
        energy = torch.bmm(Q, K.transpose(1, 2)) / (self.head_dim ** 0.5)
        attention = F.softmax(energy, dim=-1)

        # Apply attention to the values
        out = torch.bmm(attention, V)
        out = out.view(batch_size, -1)

        # Pass through the final linear layer
        out = self.fc_out(out)

        return out


class MultistrikePooling(nn.Module):
    def __init__(self, input_dim=200):
        super(MultistrikePooling, self).__init__()
        self.num_heads = 3
        self.window_sizes = [4, 8, 16, 24, 32]
        self.stride = 4
        self.input_dim = input_dim

    def forward(self, x):
        ba, de = x.shape
        fused_output01 = torch.empty((ba, 2*self.input_dim))
        fused_output1 = torch.empty((ba, self.input_dim))
        fused_output2 = torch.empty((ba, self.input_dim))
        fused_output3 = torch.empty((ba, self.input_dim))
        fused_output4 = torch.empty((ba, self.input_dim))

        fused_output01 = F.avg_pool1d(x, kernel_size=2, stride=2)
        fused_output0 = F.avg_pool1d(x, kernel_size=4, stride=4)
        fused_output1[:, :-1] = F.avg_pool1d(x, kernel_size=8, stride=4)  # [4, 8, 16, 24, 32]c(5,4)比较5种
        fused_output2[:, :-3] = F.avg_pool1d(x, kernel_size=16, stride=4)  # [1, 3, 5, 7],[-2, -7:-4, -11:-6, -15:-8]
        fused_output3[:, :-5] = F.avg_pool1d(x, kernel_size=24, stride=4)
        fused_output4[:, :-7] = F.avg_pool1d(x, kernel_size=32, stride=4)

        fused_output1[:, -1] = fused_output1[:, -2]
        fused_output2[:, -3:] = fused_output2[:, -7:-4]
        fused_output3[:, -5:] = fused_output3[:, -11:-6]
        fused_output4[:, -7:] = fused_output4[:, -15:-8]

        fused_output = torch.empty((ba, de))
        fused_output[:, 0::2] = fused_output01
        fused_output[:, 1::4] = fused_output0
        # fused_output[:, 2::4] = fused_output3
        fused_output[:, 3::4] = fused_output2

        fused_output = fused_output.cuda()
        return fused_output

# Transformer Encoder Block
class Transformer(nn.Module):
    def __init__(self, dim, model, depth=1, heads=3, mlp_dim=64, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            if model == 'MP+transFormer':
                print('MP+transFormer')
                self.layers.append(Residual(PreNorm(dim, Attention(dim, num_heads=heads))))
                self.layers.append(Residual(PreNorm(dim, MultistrikePooling(input_dim=dim//4))))  # 替换Attention要加上参数dim和num_heads=heads
                self.layers.append(Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))))
            elif model == 'noneFormer':
                print('noneFormer')
                self.layers.append(Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))))
            elif model == 'MP':
                # print('MP')
                self.layers.append(Residual(PreNorm(dim, MultistrikePooling(input_dim=dim//4))))
                self.layers.append(Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))))
            elif model == 'transFormer':
                print('transFormer')
                self.layers.append(Residual(PreNorm(dim, Attention(dim, num_heads=heads))))
                self.layers.append(Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Main Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, num_classes, in_model, cnn_m, en_m, pre_m, encoder_n, cnn_pa, input_dim, nnss, n_pool=1):
        super(TransformerModel, self).__init__()
        self.preprocess = PreprocessingModule()
        self.encoder_n = encoder_n
        self.cnn_m = cnn_m
        self.pre_m = pre_m
        self.n_pool = n_pool
        # 添加一个卷积层来降维，从 [channels, 7, 7] 到 [channels, 1, 1]
        # self.avg_pool = nn.AvgPool2d(kernel_size=(cnn_pa, cnn_pa), stride=1, padding=0)
        self.transformer_encoders = nn.ModuleList(
            [Transformer(dim=4 * input_dim, model=in_model, depth=1, heads=4, mlp_dim=32) for _ in range(encoder_n)]
        )
        self.fc = nn.Linear(input_dim//n_pool, num_classes)  # 注意修改
        # self.conv = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=(cnn_pa, cnn_pa), stride=1,
        # padding=0)
        # pe = torch.zeros(nnss+1, input_dim)
        # if en_m == 'pos':
        #     for pos in range(nnss+1):
        #         for i in range(0, input_dim, 2):
        #             pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / input_dim)))
        #             pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / input_dim)))
        # pe = pe.unsqueeze(0)
        # self.register_buffer('pe', pe)

    def forward(self, x):
        # print(f'x0  shape:{x.shape}')
        batch_size, seq_len, channels = x.size()
        x = self.preprocess.ya_preprocess_cube(x, mod=self.pre_m)  # 交叉融合 (batch,4,200)→(batch,200*4)

        for i in range(self.encoder_n):
            x = self.transformer_encoders[i](x)

        remainder = channels % self.n_pool
        divisible_channels = channels - remainder
        if remainder == 0:
            x = x.view(batch_size, channels // self.n_pool, 4 * self.n_pool).mean(dim=2).cuda()
        else:
            xt = torch.empty((batch_size, channels // self.n_pool + 1))
            xt[:, :-1] = x[:, :divisible_channels*4].view(batch_size, channels//self.n_pool, 4*self.n_pool).mean(dim=2)
            xt[:, -1] = x[:, divisible_channels*4:].mean(dim=1)
            x = xt.cuda()

        x = self.fc(x)
        # print(f'x4  shape:{x.shape}')
        return x