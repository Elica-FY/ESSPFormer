import torch
import torch.nn as nn
import torch.nn.functional as F
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

# SPPCF
class SPPCFModule(nn.Module):
    def __init__(self, input_dim=200):
        super(SPPCFModule, self).__init__()
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

        # Group Pooling in SPPCF
        fused_output01 = F.avg_pool1d(x, kernel_size=2, stride=2)
        fused_output0 = F.avg_pool1d(x, kernel_size=4, stride=4)
        fused_output1[:, :-1] = F.avg_pool1d(x, kernel_size=8, stride=4)
        fused_output2[:, :-3] = F.avg_pool1d(x, kernel_size=16, stride=4)
        fused_output3[:, :-5] = F.avg_pool1d(x, kernel_size=24, stride=4)
        fused_output4[:, :-7] = F.avg_pool1d(x, kernel_size=32, stride=4)

        fused_output1[:, -1] = fused_output1[:, -2]
        fused_output2[:, -3:] = fused_output2[:, -7:-4]
        fused_output3[:, -5:] = fused_output3[:, -11:-6]
        fused_output4[:, -7:] = fused_output4[:, -15:-8]

        # Cross Fusion in SPPCF
        fused_output = torch.empty((ba, de))
        fused_output[:, 0::2] = fused_output01
        fused_output[:, 1::4] = fused_output0
        # fused_output[:, 2::4] = fused_output3
        fused_output[:, 3::4] = fused_output2

        fused_output = fused_output.cuda()
        return fused_output

class SPPCF_encoder(nn.Module):
    def __init__(self, dim, depth=1, mlp_dim=64, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(Residual(PreNorm(dim, SPPCFModule(input_dim=dim // 4))))
            self.layers.append(Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Group Pooling in the end
class Group_Pooling(nn.Module):
    def __init__(self, channels, num_classes, groupPoolScale):
        super().__init__()
        self.channels = channels
        self.gPS = groupPoolScale
        self.remainder = channels % groupPoolScale
        self.divisible_channels = channels - self.remainder
        if self.remainder == 0:
            self.fc = nn.Linear(channels // groupPoolScale, num_classes)
        else:
            self.fc = nn.Linear(channels // groupPoolScale + 1, num_classes)

    def groupPooling(self, x):
        if self.remainder == 0:
            x = x.view(x.shape[0], self.channels // self.gPS, 4 * self.gPS).mean(dim=2).cuda()
        else:
            xt = torch.empty((x.shape[0], self.channels // self.gPS + 1))
            xt[:, :-1] = x[:, :self.divisible_channels * 4].view(x.shape[0], self.channels // self.gPS, 4 * self.gPS)\
                .mean(dim=2)
            xt[:, -1] = x[:, self.divisible_channels * 4:].mean(dim=1)
            x = xt.cuda()
        return x

# Main Model
class ESSPFormerModel(nn.Module):
    def __init__(self, num_classes, encoder_n, input_dim, groupPoolScale=4):
        super(ESSPFormerModel, self).__init__()
        self.encoder_n = encoder_n
        self.GPScale = groupPoolScale
        self.SPPCF_encoders = nn.ModuleList(
            [SPPCF_encoder(dim=4*input_dim, depth=1, mlp_dim=32) for _ in range(encoder_n)]
        )
        self.group_pooling = Group_Pooling(input_dim, num_classes, groupPoolScale)

    def forward(self, x):
        for i in range(self.encoder_n):
            x = self.SPPCF_encoders[i](x)
        x = self.group_pooling.groupPooling(x)
        x = self.group_pooling.fc(x)
        return x