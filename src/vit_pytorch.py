import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F

# 残差连接
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 层归一化
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# class DenseConnect(nn.Module):
#     def __init__(self,num_channel):
#         super().__init__()
#         self.num_channel= num_channel
#         self.net = nn.Sequential(
#             nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0)
#         )
#     def forward(self,last_input,last_output ):
#         x= torch.cat([last_input.unsqueeze(3),last_output.unsqueeze(3)], dim=3).squeeze(3)
#         x=self.net(x)
#         return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads#heads=4,dim_head=16
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask = None):
        # x:[b,n,dim][64,201,64]
        b, n, _, h = *x.shape, self.heads#n:201;b:64;_:64;h:4

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])(64,201,64)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)#v:(64,4,201,16)q:(64,4,201,16)
        #将查询（q）、键（k）、值（v）的形状调整为 [b, h, n, d]，其中 h 为头数，d 为每个头的维度。
        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale#dots:(64,4,201,201)
        #将查询（q）和键（k）张量相互关联，生成一个分数矩阵，并缩放，'i' 和 'j' 分别表示查询和键的位置序号。
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)#attn:(64,4,201,201)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)#out(64,4,201,16)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')#out(64,201,64)
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout = dropout)))
            ]))
        self.mode = mode
        # self.DenseConnect=DenseConnect(num_channel=num_channel)
        self.FlowConnect=nn.ModuleList([])
        for _ in range(depth):
            self.FlowConnect.append(nn.Conv2d(num_channel+1,num_channel+1,[1,2],1,0))
        self.skipcat = nn.ModuleList([])
        for _ in range(depth-2):
            self.skipcat.append(nn.Conv2d(num_channel+1, num_channel+1, [1, 2], 1, 0))


    def forward(self, x, mask = None):
        if self.mode == 'ViT':
            for attn, ff in self.layers:
                x = attn(x, mask = mask)
                x = ff(x)
                #print(np.shape(x))
        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:           
                last_output.append(x)
                if nl > 1:             
                    x = self.skipcat[nl-2](torch.cat([x.unsqueeze(3), last_output[nl-2].unsqueeze(3)], dim=3)).squeeze(3)
                    #将两个张量沿第三个维度拼接——>squeeze(3)将第三维度进行压缩——>拼接后的张量传递给 self.skipcat[nl-2](卷积层)
                x = attn(x, mask = mask)
                x = ff(x)
                nl += 1
        elif self.mode=='FC':
            last_output=[]
            last_input=[x,]
            nl = 0
            for attn,ff in self.layers:
                x = attn(x, mask=mask)
                x = ff(x)
                last_output.append(x)
                x = self.FlowConnect[nl](torch.cat([last_input[nl].unsqueeze(3),last_output[nl].unsqueeze(3)],dim=3)).squeeze(3)
                #unsqueeze(3) 的作用是在第3维上增加一个维度，将原来的维度（64，201，64）变成了新的维度（64，201，64，1）,确保张良能够沿第三维度进行拼接
                last_input.append(x)
                nl += 1
            #各层融合输出
            x = torch.mean(torch.stack(last_output, dim=-1), dim=-1)#注意cat与stack的区别
            #x = last_output[-1]
        return x #transformer模型最后的输出


# --------------------ViT input parameters----------------------
#     image_size = args.patches,        e.g. 7*7的像素点作为input
#     near_band = args.band_patches,    e.g. 3个光谱值为一组进行组合
#     num_patches = band,               光谱通道数
#     num_classes = num_classes,        样本总类别数
#     dim = 64,                         batchSize
#     depth = 5,                        transformer block num
#     heads = 4,                        multi-head num
#     mlp_dim = 8,
#     dropout = 0.1,
#     emb_dropout = 0.1,
#     mode = args.mode
# --------------------------------------------------------------
class ViT(nn.Module):
    def __init__(self, image_size, near_band, num_patches, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=1, dim_head = 16, dropout=0., emb_dropout=0., mode='ViT'):
        super().__init__()
        # 输入一共多少个光谱点
        patch_dim = image_size ** 2 * near_band
        # 定义一个随机矩阵，作为nn.Module中的可训练参数使用
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # 定义全连接层，输入尺寸patch_dim，输出尺寸dim
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        # 定义一个随机矩阵，作为nn.Module中的可训练参数使用
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # 随机将部分神经元的输出置为0来减少过拟合。在测试时（model.eval模式时）并不会使用Dropout
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, mode)

        self.pool = pool
        # torch.nn.Identity( ) 相当于一个恒等函数 f(x)=x,常用于替代最后一层的全连接网络
        self.to_latent = nn.Identity()
        # 定义多层感知器 = 层归一化 + 全连接层
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    def forward(self, x, mask = None):
       
        # patchs[batch, patch_num, patch_size*patch_size*c]  [batch,200,145*145]
        # x = rearrange(x, 'b c h w -> b c (h w)')

        ## embedding every patch vector to embedding size: [batch, patch_num, embedding_size]
        x = self.patch_to_embedding(x) #[b,n,dim]
        b, n, _ = x.shape

        # add position embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b) #[b,1,dim]
        x = torch.cat((cls_tokens, x), dim = 1) #[b,n+1,dim]
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x = self.transformer(x, mask)

        # classification: using cls_token output
        x = self.to_latent(x[:,0])
        # 将Transformer的输出中的第一个位置（通常是CLS令牌的输出）提取出来
        # MLP classification layer
        return self.mlp_head(x)

if __name__ == "__main__":
    a=0