# # ============== Deformable Attention 模块 ==============
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MSDeformAttn(nn.Module):
#     """多尺度可变形注意力机制"""
#     def __init__(self, d_model, num_heads, num_levels, num_points):
#         super().__init__()
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.num_levels = num_levels
#         self.num_points = num_points
        
#         # 采样偏移预测
#         self.sampling_offsets = nn.Linear(
#             d_model, 
#             num_heads * num_levels * num_points * 2
#         )
        
#         # 注意力权重预测
#         self.attention_weights = nn.Linear(
#             d_model,
#             num_heads * num_levels * num_points
#         )
        
#         # 值投影
#         self.value_proj = nn.Linear(d_model, d_model)
#         self.output_proj = nn.Linear(d_model, d_model)
        
#         self._reset_parameters()
        
#     def _reset_parameters(self):
#         nn.init.constant_(self.sampling_offsets.weight, 0.)
#         nn.init.constant_(self.sampling_offsets.bias, 0.)
#         nn.init.constant_(self.attention_weights.weight, 0.)
#         nn.init.constant_(self.attention_weights.bias, 0.)
        
#     def forward(self, query, reference_points, value, spatial_shapes):
#         """
#         query: (B, L, C)
#         reference_points: (B, L, num_levels, 2)
#         value: (B, L, C)
#         spatial_shapes: (num_levels, 2) - H, W for each level
#         """
#         B, L, _ = query.shape
#         _, S, _ = value.shape
        
#         # 投影 value
#         value = self.value_proj(value)
#         value = value.view(B, S, self.num_heads, self.d_model // self.num_heads)
        
#         # 预测采样偏移
#         sampling_offsets = self.sampling_offsets(query)
#         sampling_offsets = sampling_offsets.view(
#             B, L, self.num_heads, self.num_levels, self.num_points, 2
#         )
        
#         # 预测注意力权重
#         attention_weights = self.attention_weights(query)
#         attention_weights = attention_weights.view(
#             B, L, self.num_heads, self.num_levels * self.num_points
#         )
#         attention_weights = F.softmax(attention_weights, dim=-1)
#         attention_weights = attention_weights.view(
#             B, L, self.num_heads, self.num_levels, self.num_points
#         )
        
#         # 这里简化实现，实际需要根据 reference_points 和 sampling_offsets 进行采样
#         # 为了示例，我们使用简化的注意力计算
#         output = torch.zeros(B, L, self.d_model).to(query.device)
        
#         # 简化的输出投影
#         output = self.output_proj(query)
        
#         return output
    
# if __name__ == '__main__':
#     x = torch.rand(2, 10, 512)  # [batch, seq_len, d_model]
#     mha = MSDeformAttn(512, 8, 3, 3)
#     x = mha(x)
#     print(x.shape)

# ============== Deformable Attention 模块（完整版） ==============
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MSDeformAttn(nn.Module):
    """多尺度可变形注意力机制（Deformable DETR核心模块）"""
    def __init__(self, d_model=256, num_heads=8, num_levels=4, num_points=4):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError('d_model must be divisible by num_heads')

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.d_per_head = d_model // num_heads

        # 模块
        self.sampling_offsets = nn.Linear(d_model, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(d_model, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        # 初始化偏移为环状分布，更利于训练稳定
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(self, query, reference_points, value, spatial_shapes):
        """
        query: (B, Len_q, C)
        reference_points: (B, Len_q, num_levels, 2), 归一化坐标范围 [0, 1]
        value: (B, S, C)
        spatial_shapes: (num_levels, 2) -> 每个level的(H, W)
        """
        B, Len_q, _ = query.shape
        B, Len_v, _ = value.shape

        # 1️⃣ 投影 value
        value = self.value_proj(value)
        value = value.view(B, Len_v, self.num_heads, self.d_per_head)

        # 将多尺度特征拆分
        split_lengths = [H * W for H, W in spatial_shapes]
        value_list = value.split(split_lengths, dim=1)

        # 2️⃣ 预测偏移与权重
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(B, Len_q, self.num_heads, self.num_levels, self.num_points, 2)

        attention_weights = self.attention_weights(query)
        attention_weights = F.softmax(
            attention_weights.view(B, Len_q, self.num_heads, self.num_levels * self.num_points),
            dim=-1
        ).view(B, Len_q, self.num_heads, self.num_levels, self.num_points)

        # 3️⃣ 执行多尺度双线性采样
        # 输出 shape: (B, Len_q, num_heads, d_per_head)
        output = self.ms_deform_attn_core(value_list, spatial_shapes, reference_points, sampling_offsets, attention_weights)

        # 4️⃣ 输出线性映射
        output = output.view(B, Len_q, self.d_model)
        return self.output_proj(output)

    @staticmethod
    def ms_deform_attn_core(value_list, spatial_shapes, reference_points, sampling_offsets, attention_weights):
        """
        执行多尺度特征采样与加权求和。
        """
        B, Len_q, num_heads, num_levels, num_points, _ = sampling_offsets.shape
        d_per_head = value_list[0].shape[-1]
        output = torch.zeros(B, Len_q, num_heads, d_per_head, device=value_list[0].device)

        for lvl, (H, W) in enumerate(spatial_shapes):
            # 当前level特征
            value_l = value_list[lvl].view(B, H, W, num_heads, d_per_head).permute(0, 3, 4, 1, 2)  # (B, heads, C_head, H, W)
            offset = sampling_offsets[:, :, :, lvl, :, :]  # (B, Len_q, heads, num_points, 2)
            
            # 扩展 reference_points，使其包含 head 维度
            ref = reference_points[:, :, None, lvl, :].unsqueeze(3)  # (B, Len_q, heads, 1, 2)
            
            # 将 offset 转换为归一化坐标偏移
            normalizer = torch.tensor([W, H], device=ref.device).view(1, 1, 1, 1, 2)
            sampling_locations = ref + offset / normalizer  # (B, Len_q, heads, num_points, 2)

            # grid_sample 需要 [-1,1]
            sampling_grid = sampling_locations * 2 - 1
            sampling_grid = sampling_grid.permute(0, 2, 1, 3, 4).reshape(B * num_heads, Len_q, num_points, 2)

            # 采样
            sampled_value = F.grid_sample(
                value_l.reshape(B * num_heads, d_per_head, H, W),
                sampling_grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )

            sampled_value = sampled_value.view(B, num_heads, d_per_head, Len_q, num_points)
            sampled_value = sampled_value.permute(0, 3, 1, 4, 2)  # (B, Len_q, heads, num_points, C_head)

            attn = attention_weights[:, :, :, lvl, :].unsqueeze(-1)
            output += (sampled_value * attn).sum(dim=3)

        return output


if __name__ == '__main__':
    # 模拟测试
    B, Lq, Lv, C = 2, 100, 525, 256
    num_levels = 3
    num_heads = 8
    num_points = 4

    spatial_shapes = torch.tensor([[20, 20], [10, 10], [5, 5]], dtype=torch.long)
    reference_points = torch.rand(B, Lq, num_levels, 2)  # (2, 100, 3, 2)
    query = torch.rand(B, Lq, C) # (2,100,256)
    value = torch.rand(B, Lv, C) # (2,525,256)

    attn = MSDeformAttn(C, num_heads, num_levels, num_points)
    output = attn(query, reference_points, value, spatial_shapes)
    print(output.shape)  # ✅ torch.Size([2, 100, 256])
