# ============== Encoder 模块 ==============
import torch
import torch.nn as nn
import torch.nn.functional as F
from Deformable_DETR_models.MSDeformAttn import MSDeformAttn
from Deformable_DETR_models.FFN import FeedForward

class EncoderLayer(nn.Module):
    """Transformer Encoder 层"""
    def __init__(self, d_model, num_heads, num_levels, num_points, d_ff, dropout=0.1):
        super().__init__()
        self.MSDeformAttn = MSDeformAttn(d_model, num_heads, num_levels, num_points)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        
    def forward(self, query, reference_points, value, spatial_shapes, level_start_index):
        # 注意力 + Add & Norm
        output = self.MSDeformAttn(query, reference_points, value, spatial_shapes)
        query = value + self.norm1(output)

        # FFN + Add & Norm
        ffn_output = self.ffn(query)
        query = query + self.dropout1(ffn_output)
        query = self.norm2(query)
        
        return query


class TransformerEncoder(nn.Module):
    """Transformer Encoder"""
    def __init__(self, num_layers, d_model, num_heads, num_levels, num_points, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, num_levels, num_points, d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index):
        query = src + pos
        value = src
        for layer in self.layers:
            query = layer(query, reference_points, value, spatial_shapes, level_start_index)
        return query


if __name__ == '__main__':
    # 模拟测试
    B, Lq, Lv, C = 2, 100, 525, 256
    num_levels = 3
    num_heads = 8
    num_points = 4
    d_ff = 2048

    spatial_shapes = torch.tensor([[20, 20], [10, 10], [5, 5]], dtype=torch.long)
    reference_points = torch.rand(B, Lq, num_levels, 2)  # (2, 100, 3, 2)
    query = torch.rand(B, Lq, C) # (2,100,256)
    value = torch.rand(B, Lv, C) # (2,525,256)

    encoder = TransformerEncoder(6, C, num_heads, num_levels, num_points, d_ff)
    output = encoder(query, reference_points, value, spatial_shapes)
    print(output.shape)