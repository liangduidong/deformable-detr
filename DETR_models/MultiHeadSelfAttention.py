import torch
import torch.nn as nn
import math
import torch.nn.functional as F

"""多头注意力机制"""
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # 32
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换并分头
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2) # [2, 8, 10, 64] 
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)   
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) # [2, 8, 10, 10]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力权重
        context = torch.matmul(attn, V) # [2, 8, 10, 64]
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context) # [2, 10, 512]
        
        return output, attn


if __name__ == '__main__':
    mha = MultiHeadSelfAttention(256, 8)
    x = torch.rand(2, 64, 256) # [batch, seq_len, d_model]
    out, attn = mha(x, x, x)
    print(out.shape, attn.shape)