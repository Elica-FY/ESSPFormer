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

# cross-fusion in SSGPCF
class CFModule:
    def __init__(self, correlation_threshold=0.998):
        self.correlation_threshold = correlation_threshold

    def cross_fuse(self, cube):
        batch_size, n_sp, spectral_dim = cube.shape
        selected_spectra = cube
        cross_fused_spectra = torch.empty((batch_size, 4 * spectral_dim), device=cube.device)

        cross_fused_spectra[:, 0::4] = torch.min(selected_spectra[:, :-1, :], dim=1).values
        cross_fused_spectra[:, 1::4] = torch.mean(selected_spectra[:, :-1, :], dim=1)
        cross_fused_spectra[:, 2::4] = selected_spectra[:, -1, :]
        cross_fused_spectra[:, 3::4] = torch.max(selected_spectra[:, :-1, :], dim=1).values  # A[:-2,:,:]

        cross_fused_spectra.cuda()
        return cross_fused_spectra

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

        fused_output = torch.empty((ba, de))
        fused_output[:, 0::2] = fused_output01
        fused_output[:, 1::4] = fused_output0
        # fused_output[:, 2::4] = fused_output3
        fused_output[:, 3::4] = fused_output2

        fused_output = fused_output.cuda()
        return fused_output

class SPPCFformer(nn.Module):
    def __init__(self, dim, depth=1, heads=3, mlp_dim=64, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(Residual(PreNorm(dim, SPPCFModule(input_dim=dim // 4))))
            self.layers.append(Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Main Model
class ESSPFormerModel(nn.Module):
    def __init__(self, num_classes, encoder_n, input_dim, nnss, n_pool=1):
        super(ESSPFormerModel, self).__init__()
        self.CF = CFModule()
        self.encoder_n = encoder_n
        self.n_pool = n_pool
        self.SPPCFformer_encoders = nn.ModuleList(
            [SPPCFformer(dim=4 * input_dim, depth=1, heads=4, mlp_dim=32) for _ in range(encoder_n)]
        )
        self.fc0 = nn.Linear(input_dim // n_pool, num_classes)
        self.fc1 = nn.Linear(input_dim//n_pool + 1, num_classes)

    def forward(self, x):
        batch_size, seq_len, channels = x.size()
        x = self.CF.cross_fuse(x)
        for i in range(self.encoder_n):
            x = self.SPPCFformer_encoders[i](x)

        remainder = channels % self.n_pool
        divisible_channels = channels - remainder
        if remainder == 0:
            x = x.view(batch_size, channels // self.n_pool, 4 * self.n_pool).mean(dim=2).cuda()
            x = self.fc0(x)
        else:
            xt = torch.empty((batch_size, channels // self.n_pool + 1))
            xt[:, :-1] = x[:, :divisible_channels*4].view(batch_size, channels//self.n_pool, 4*self.n_pool).mean(dim=2)
            xt[:, -1] = x[:, divisible_channels*4:].mean(dim=1)
            x = xt.cuda()
            x = self.fc1(x)
        return x