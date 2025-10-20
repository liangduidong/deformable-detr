# ============== Encoder 模块 ==============
import torch
import torch.nn as nn
import torch.nn.functional as F
from .MultiHeadSelfAttention import MultiHeadSelfAttention
from .FFN import FeedForward

class EncoderLayer(nn.Module):
    """Transformer Encoder 层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, res, mask=None):
        # 自注意力 + Add & Norm
        attn_output, _ = self.self_attn(x, x, res, mask)
        x = res + self.norm1(attn_output)
        
        # FFN + Add & Norm
        ffn_output = self.ffn(x)
        x = x + self.norm2(ffn_output)
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder"""
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, res, mask=None):
        for layer in self.layers:
            x = layer(x, res, mask)
            res = x
        return x


if __name__ == '__main__':
    x = torch.rand(2, 64, 256)
    B, L, C = x.shape
    encoder = TransformerEncoder(num_layers=6, d_model=C, num_heads=8, d_ff=2048)
    out = encoder(x, x) # [B, L, C]
    print(out.shape)
