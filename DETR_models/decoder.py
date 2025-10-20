# ============== Decoder 模块 ==============
import torch
import torch.nn as nn
from .MultiHeadSelfAttention import MultiHeadSelfAttention
from .FFN import FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_queries, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x, res, memory, tgt_mask=None, memory_mask=None):
        # 自注意力
        attn_output, _ = self.self_attn(x, x, x, mask=tgt_mask)
        x = res + self.norm1(attn_output)

        # 交叉注意力
        cross_output, _ = self.cross_attn(x, memory, memory, mask=memory_mask)
        x = x + self.norm2(cross_output)

        # FFN
        ffn_output = self.dropout1(self.ffn(x))
        x = x + self.norm3(ffn_output)

        return x



class TransformerDecoder(nn.Module):
    """Transformer Decoder"""
    def __init__(self, num_layers, d_model, num_heads, d_ff, num_queries, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, num_queries, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, res, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            x = layer(x, res, memory)
            res = x
        return x

if __name__ == "__main__":
    x = torch.rand(2, 100, 256)
    B, query, C = x.shape
    decoder = TransformerDecoder(num_layers=6, d_model=C, num_heads=8, d_ff=2048, num_queries=query)
    memory = torch.rand(2, 64, 256)
    out = decoder(x, memory) 
    print(out.shape)