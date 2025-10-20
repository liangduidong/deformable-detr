# ============== Deformable DETR 完整模型 ==============
from anyio import value
import torch
import torch.nn as nn
import torch.nn.functional as F
from Deformable_DETR_models.backbone import resnet50_multiscale
from Deformable_DETR_models.encoder import TransformerEncoder
from Deformable_DETR_models.decoder import TransformerDecoder
from utils.position_encoding import LearnablePositionEmbedding, LearnableQueryPos
from utils.help import get_reference_points, get_level_start_index
class DeformableDETR(nn.Module):
    """Deformable DETR: Deformable Detection Transformer"""
    def __init__(
        self,
        num_classes,
        d_model=256,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_heads=8,
        d_ff=2048,
        num_queries=300,
        num_levels=3,
        num_points=4,
        dropout=0.1
    ):
        super().__init__()
         # ========== 修复1: 添加所有必要的类属性 ==========
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_queries = num_queries
        self.num_levels = num_levels
        self.num_points = num_points
        self.dropout = dropout
        # 多尺度 Backbone
        self.backbone = resnet50_multiscale()
        # 用于将不同尺度特征投影到统一维度
        backbone_channels = [512, 1024, 2048]  # 对应 return_layers=[1,2,3]

        self.input_proj = nn.ModuleList([
                            nn.Conv2d(backbone_channels[i], d_model, kernel_size=1)
                          for i in range(min(num_levels, len(backbone_channels)))
])

        # Transformer Encoder
        self.encoder = TransformerEncoder(
            num_encoder_layers, d_model, num_heads, num_levels, num_points, d_ff, dropout=dropout
        )
        # 可学习位置编码只初始化一次
        self.encoder_pos_emb = LearnablePositionEmbedding(num_pos_feats=d_model//2)

        # Object Queries (可学习的查询向量)
        self.object_queries = nn.Embedding(num_queries, d_model)
        nn.init.uniform_(self.object_queries.weight)

        # Reference Points (参考点)
        self.reference_points = nn.Linear(d_model, 2)
        
        # Decoder with Deformable Attention
        self.decoder = TransformerDecoder(
            num_decoder_layers, d_model, num_heads,  num_levels, num_points, d_ff, dropout
        )
        self.decoder_pos_emb = LearnableQueryPos(num_queries=num_queries, d_model=d_model)
        
        # 预测头
        self.class_head = nn.Linear(d_model, num_classes + 1)
        self.bbox_head = MLP(d_model, d_model, 4, 3)  # 3层MLP
        self._reset_parameters()
    
    def _reset_parameters(self):
        """初始化参数"""
        # 初始化投影层
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj.weight, gain=1)
            nn.init.constant_(proj.bias, 0)
        
        # 初始化参考点预测层
        nn.init.constant_(self.reference_points.weight, 0.)
        nn.init.constant_(self.reference_points.bias, 0.)
        
        # 初始化分类头
        nn.init.constant_(self.class_head.bias, 0.)
        
    def forward(self, images):
        B = images.size(0)
        device = images.device
        
        # ========== 1. 多尺度特征提取 ==========
        # 从 backbone 获取多尺度特征
        backbone_features = self.backbone(images)  # 返回多个尺度的特征

        # 投影到统一维度并添加位置编码
        multi_scale_features = []
        spatial_shapes = []

        for lvl, feat in enumerate(backbone_features[:self.num_levels]):
            # 投影到 d_model 维度
            feat = self.input_proj[lvl](feat)  # [B, d_model, H, W]
            
            B, C, H, W = feat.shape
            spatial_shapes.append([H, W])
            
            # 添加位置编码
            pos_embed = self.encoder_pos_emb(feat)  # [B, d_model, H, W]
            
            # 展平: [B, d_model, H, W] -> [B, H*W, d_model]
            feat_flat = feat.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            pos_flat = pos_embed.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            
            multi_scale_features.append((feat_flat, pos_flat))
        
        # 连接所有尺度的特征
        src_flatten = torch.cat([feat for feat, _ in multi_scale_features], dim=1)  # [B, L, C]
        pos_flatten = torch.cat([pos for _, pos in multi_scale_features], dim=1)  # [B, L, C]
        
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=device)
        level_start_index = get_level_start_index(spatial_shapes)
        
        # ========== 2. Encoder ==========
        # 生成参考点
        reference_points_encoder = get_reference_points(spatial_shapes, device)
        reference_points_encoder = reference_points_encoder[None].repeat(B, 1, 1, 1)  # [B, L, num_levels, 2]
        
        # Encoder 前向传播
        memory = self.encoder(
            src=src_flatten,
            pos=pos_flatten,
            reference_points=reference_points_encoder,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        )
        
        # ========== 3. Decoder ==========
        # Object Queries
        query_embed = self.object_queries.weight  # [num_queries, d_model]
        query = torch.zeros_like(query_embed).unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, d_model]
        
        # Query 位置编码
        query_pos = self.decoder_pos_emb(query)  # [B, num_queries, d_model]
        
        # 预测参考点
        reference_points_unlevel  = self.reference_points(query_embed).sigmoid()  # [num_queries, 2]
        reference_points_unlevel  = reference_points_unlevel .unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, 2]
        reference_points_decoder = reference_points_unlevel .unsqueeze(2).repeat(1, 1, self.num_levels, 1)  # [B, num_queries, num_levels, 2]
        
        # Decoder 前向传播
        hs = self.decoder(
            obj_queries=query,
            query_pos=query_pos,
            reference_points=reference_points_decoder,
            memory=memory,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        )
        
        # ========== 4. 预测头 ==========
        # 使用最后一层 decoder 的输出
        outputs_class = self.class_head(hs)  # [B, num_queries, num_classes+1]
        
         # 边界框预测（相对于参考点的偏移）
        tmp = self.bbox_head(hs)  # 偏移
        reference_unsigmoid = inverse_sigmoid(reference_points_unlevel)
        tmp[..., :2] += reference_unsigmoid
        outputs_coord = tmp.sigmoid()
        
        return {
            'class_logits': outputs_class,
            'bbox_pred': outputs_coord
        }

# ============== MLP 辅助模块 ==============

class MLP(nn.Module):
    """简单的多层感知机"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x     

# ============== 辅助函数 ==============

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

# ============== 测试代码 ==============

if __name__ == "__main__":
    print("=" * 60)
    print("Deformable DETR 完整实现测试")
    print("=" * 60)
    
    # 配置
    num_classes = 80
    num_queries = 300
    batch_size = 2
    
    # 创建简化版模型（用于测试）
    print("\nDeformable DETR 模型...")
    model = DeformableDETR(num_classes=num_classes)
    
    # 测试输入
    dummy_images = torch.randn(batch_size, 3, 640, 640)
    print(f"输入图像形状: {dummy_images.shape}")
    
    # 前向传播
    print("\n执行前向传播...")
    try:
        with torch.no_grad():
            outputs = model(dummy_images)
        
        print("\n✓ 前向传播成功！")
        print(f"\n输出:")
        print(f"  类别预测形状: {outputs['class_logits'].shape}")  # [B, num_queries, num_classes+1]
        print(f"  边界框预测形状: {outputs['bbox_pred'].shape}")  # [B, num_queries, 4]
        
        # 统计信息
        print(f"\n统计信息:")
        print(f"  类别预测范围: [{outputs['class_logits'].min():.3f}, {outputs['class_logits'].max():.3f}]")
        print(f"  边界框预测范围: [{outputs['bbox_pred'].min():.3f}, {outputs['bbox_pred'].max():.3f}]")
        
        # 参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n模型参数:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        
    except Exception as e:
        print(f"\n✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
    