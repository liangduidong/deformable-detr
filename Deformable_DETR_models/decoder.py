# ============== Decoder 模块 ==============
import torch
import torch.nn as nn
from Deformable_DETR_models.MultiHeadSelfAttention import MultiHeadSelfAttention
from Deformable_DETR_models.MSDeformAttn import MSDeformAttn
from Deformable_DETR_models.FFN import FeedForward

class DecoderLayer(nn.Module):
    """Transformer Decoder 层"""
    def __init__(self, d_model, num_heads,  num_levels, num_points, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.MSDeformAttn = MSDeformAttn(d_model, num_heads, num_levels, num_points)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        
    def forward(self, obj_queries, query_pos, reference_points, value, spatial_shapes, level_start_index):
        # 自注意力 + Add & Norm
        res = obj_queries
        obj_queries = obj_queries + query_pos
        attn_output, _ = self.self_attn(obj_queries, obj_queries, res)
        queries = res + self.norm1(attn_output)
        
        # deformable注意力 + Add & Norm
        output = self.MSDeformAttn(queries, reference_points, value, spatial_shapes)
        queries = queries + self.norm2(output)
        
        # FFN + Add & Norm
        ffn_output = self.ffn(queries)
        queries = queries + self.norm3(self.dropout1(ffn_output))
        
        return queries


class TransformerDecoder(nn.Module):
    """Transformer Decoder"""
    def __init__(self, num_layers, d_model, num_heads, num_levels, num_points, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads,  num_levels, num_points, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, obj_queries, query_pos, reference_points, memory, spatial_shapes, level_start_index):
        for layer in self.layers:
            obj_queries = layer(obj_queries, query_pos, reference_points, memory, spatial_shapes, level_start_index)
        return obj_queries

if __name__ == "__main__":
    # 模拟测试
    B, Lq, Lv, C = 2, 100, 525, 256
    num_levels = 3
    num_heads = 8
    num_points = 4
    d_ff = 2048

    spatial_shapes = torch.tensor([[20, 20], [10, 10], [5, 5]], dtype=torch.long)
    reference_points = torch.rand(B, Lq, num_levels, 2)  # (2, 100, 3, 2)
    obj_queries = torch.rand(B, Lq, C) # (2,100,256)
    value = torch.rand(B, Lv, C) # (2,525,256)

    decoder = TransformerDecoder(6, C, num_heads, num_levels, num_points, d_ff)
    output = decoder(obj_queries, reference_points, value, spatial_shapes)
    print(output.shape)