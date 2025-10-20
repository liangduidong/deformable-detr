import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


# ============== 匈牙利匹配器 ==============

class HungarianMatcher(nn.Module):
    """
    匈牙利匹配算法：将预测结果与真实标签进行最优匹配
    """
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        outputs: dict with 'class_logits' [B, num_queries, num_classes+1]
                            'bbox_pred' [B, num_queries, 4]
        targets: list of dict, each with 'labels' [num_targets] and 'boxes' [num_targets, 4]
        
        Returns:
            list of (pred_idx, target_idx) tuples for each image
        """
        bs, num_queries = outputs["class_logits"].shape[:2]
        
        # 展平以进行批量计算
        out_prob = outputs["class_logits"].flatten(0, 1).softmax(-1)  # [B*num_queries, num_classes+1]
        out_bbox = outputs["bbox_pred"].flatten(0, 1)  # [B*num_queries, 4]
        
        # 连接所有目标标签和框
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        
        # 计算分类成本
        cost_class = -out_prob[:, tgt_ids]
        
        # 计算 L1 成本
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # 计算 GIoU 成本
        cost_giou = -self.generalized_box_iou(
            self.box_cxcywh_to_xyxy(out_bbox),
            self.box_cxcywh_to_xyxy(tgt_bbox)
        )
        
        # 最终成本矩阵
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        
        # 为每个图像执行匈牙利匹配
        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        for i, c in enumerate(C.split(sizes, -1)):
            # c: [num_queries, num_targets_i]
            indices.append(linear_sum_assignment(c[i]))
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) 
                for i, j in indices]
    
    @staticmethod
    def box_cxcywh_to_xyxy(x):
        """中心格式转换为角点格式"""
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)
    
    @staticmethod
    def generalized_box_iou(boxes1, boxes2):
        """计算 GIoU"""
        # 确保框的格式正确
        assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
        
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # 计算交集
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
        
        # 计算并集
        union = area1[:, None] + area2 - inter
        
        # 计算 IoU
        iou = inter / union
        
        # 计算最小外接矩形
        lti = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rbi = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
        whi = (rbi - lti).clamp(min=0)  # [N,M,2]
        areai = whi[:, :, 0] * whi[:, :, 1]
        
        # GIoU
        return iou - (areai - union) / areai