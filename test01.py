import numpy as np
from itertools import combinations
import torch
import math
import torch.nn as nn

def f(a, b):
    # 生成从 0 到 a-1 的列表
    elements = list(range(a))
    # 获取所有组合
    comb_array = list(combinations(elements, b))
    # 转换为 NumPy 数组
    result = np.append(comb_array, np.full((np.shape(comb_array)[0], 1), a - 1), axis=1)
    return np.array(result)
# # 示例
# result = f(7, 4)
# print(result)
# print(result.shape)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model

        # 根据 pos 和 i 创建一个常量 PE 矩阵
        pe = torch.zeros(max_seq_len, d_model)
        print(pe.shape)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        # 增加 batch 维度
        pe = pe.unsqueeze(0)
        print(pe.shape)

        # 将 pe 注册为 buffer，不作为模型的参数，但会在模型中使用
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 使得词嵌入表示相对大一些
        x = x * math.sqrt(self.d_model)

        # 获取输入序列的长度
        seq_len = x.size(1)

        # 增加位置编码到词嵌入表示中，不需要梯度
        x = x + self.pe[:, :seq_len].cuda()

        return x

pos_encoder = PositionalEncoder(d_model=512, max_seq_len=100)
pos_encoder.cuda()
x = torch.zeros(32, 50, 512).cuda()  # 一个 batch 的输入，大小为 (batch_size, seq_len, d_model)
output = pos_encoder(x)  # 加入位置编码
print(output.shape)
print(output[:,1,0])
patches = 7
n = 13
top_coords = np.concatenate([np.random.randint(0, patches, size=(n - 1, 2)), np.ones((1, 2)) + patches//2], axis=0)
print(top_coords)


