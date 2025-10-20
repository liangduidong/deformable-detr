# ============== DETR 完整模型 ==============
import torch
import torch.nn as nn
from DETR_models.encoder import TransformerEncoder
from DETR_models.decoder import TransformerDecoder
from utils.position_encoding import LearnablePositionEmbedding, LearnableQueryPos
# from matcher import HungarianMatcher
from DETR_models.backbone import resnet50

class DETR(nn.Module):
    """DETR: Detection Transformer"""
    def __init__(
        self, 
        num_classes,
        d_model=256,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_heads=8,
        d_ff=2048,
        num_queries=100,
        dropout=0.1
    ):
        super().__init__()
        
        # Backbone (这里使用resnet50)
        self.backbone = resnet50(num_classes=num_classes, out_channels=d_model)

        # Transformer Encoder
        self.encoder = TransformerEncoder(
            num_encoder_layers, d_model, num_heads, d_ff, dropout=dropout
        )
        # 可学习位置编码只初始化一次
        self.encoder_pos_emb = LearnablePositionEmbedding(num_pos_feats=d_model//2)
        
        # Object Queries (可学习的查询向量)
        self.object_queries = nn.Embedding(num_queries, d_model)
        nn.init.uniform_(self.object_queries.weight)

        # Transformer Decoder
        self.decoder = TransformerDecoder(
            num_decoder_layers, d_model, num_heads, d_ff, num_queries=num_queries, dropout=dropout
        )
        self.decoder_pos_emb = LearnableQueryPos(num_queries=num_queries, d_model=d_model)
        
        # 预测头
        self.class_head = nn.Linear(d_model, num_classes + 1)  # +1 for background
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4)  # x, y, w, h
        )
        
    def forward(self, images):
        # 图像特征提取
        features = self.backbone(images)
        B = features.shape[0]

        # Encoder
        # 位置编码
        features_pos = self.encoder_pos_emb(features)
        # 扁平化
        features = features.flatten(2).permute(0, 2, 1)  # [B, L, C], L=H*W
        features_pos = features_pos.flatten(2).permute(0, 2, 1)
        res = features
        features = features + features_pos
        memory = self.encoder(features, res)
        
        # Object Queries
        queries = self.object_queries.weight.unsqueeze(0).repeat(B, 1, 1)  # (B, num_queries, C)
        res = queries # 保存残差
        queries = queries + self.decoder_pos_emb(queries) # [B, num_queries, d_model]

        # Decoder
        decoder_output = self.decoder(queries, res, memory)
        
        # 预测类别和边界框
        class_logits = self.class_head(decoder_output)
        bbox_pred = self.bbox_head(decoder_output).sigmoid()
        
        return {
            'class_logits': class_logits,
            'bbox_pred': bbox_pred
        }
    
if __name__ == "__main__":
    dummy_images = torch.randn(2, 3, 224, 224)
    B, C, H, W = dummy_images.shape
    detr = DETR(num_classes=80,
                d_model=128,
                num_encoder_layers=6,
                num_decoder_layers=6,
                num_heads=8,
                d_ff=2048,
                num_queries=100,
                dropout=0.1)
    output = detr(dummy_images)
    print(f"Class logits shape: {output['class_logits'].shape}")  # (B, 100, 81)
    print(f"BBox predictions shape: {output['bbox_pred'].shape}")  # (B, 100, 4)