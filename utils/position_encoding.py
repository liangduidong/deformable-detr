import torch
import torch.nn as nn
import math

class PositionEmbeddingSine(nn.Module):
    """
    二维正弦位置编码，用于图像特征 (H, W)。
    参考自 DETR 官方实现。
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, mask):
        """
        mask: (B, H, W) — padding mask, 值为 False 的地方是有效区域
        """
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos  # (B, 2*num_pos_feats, H, W)

class LearnablePositionEmbedding(nn.Module):
    """
    2D 可学习位置编码
    输出形状与输入特征图相同: [B, C, H, W]
    """
    def __init__(self, num_pos_feats=256, height=100, width=100):
        super().__init__()
        # 两个可学习参数矩阵（行、列方向）
        self.row_embed = nn.Embedding(height, num_pos_feats)
        self.col_embed = nn.Embedding(width, num_pos_feats)

        # 参数初始化（遵循标准正态）
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        """
        x: [B, C, H, W] 输入特征图，仅用于确定尺寸 (H, W)
        返回: [B, C, H, W] 可学习位置编码
        """
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        # col_embed: [W, C], row_embed: [H, C]
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)

        # 拼接成 [H, W, 2*C]
        pos = torch.cat([
            y_emb[:, None, :].expand(h, w, -1),
            x_emb[None, :, :].expand(h, w, -1),
        ], dim=-1)

        # 转换为 [B, 2*C, H, W]
        pos = pos.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        return pos

class LearnableQueryPos(nn.Module):
    """
    一维可学习位置编码，用于 Decoder 的 object queries
    输入: [B, num_queries, d_model]
    输出: [B, num_queries, d_model]
    """
    def __init__(self, num_queries, d_model):
        super().__init__()
        self.pos_embed = nn.Embedding(num_queries, d_model)
        nn.init.uniform_(self.pos_embed.weight)

    def forward(self, x):
        # x: [B, num_queries, d_model]
        B, N, C = x.shape
        pos = self.pos_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, N, C]
        return pos



# =============== 测试样例 ===============
if __name__ == "__main__":
    # encoder
    B, C, H, W = 2, 256, 8, 8
    x = torch.randn(B, C, H, W)

    pos_encoder = LearnablePositionEmbedding(num_pos_feats=C // 2, height=H, width=W)
    pos = pos_encoder(x)

    print("Input shape:", x.shape)
    print("Positional encoding shape:", pos.shape)

    # 加上位置编码
    x = x + pos
    print("Output shape:", x.shape)

    # decoder 
    B, num_queries, d_model = 2, 100, 256
    queries = torch.randn(B, num_queries, d_model)
    query_pos = LearnableQueryPos(num_queries=num_queries, d_model=d_model)
    pos = query_pos(queries)
    x = queries + pos
    print(x.shape)  # [2, 100, 256]

