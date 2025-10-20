import torch
import torch.nn as nn
import torch.nn.functional as F
from .match import HungarianMatcher  

# ============== DETR 损失函数 ==============
class DETRLoss(nn.Module):
    def __init__(self, num_classes, weight_dict, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(cost_class=weight_dict['loss_ce'], 
                                        cost_bbox=weight_dict['loss_bbox'], 
                                        cost_giou=weight_dict['loss_giou']) # matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        
        self.empty_weight = torch.ones(self.num_classes + 1)
        self.empty_weight[-1] = self.eos_coef
        
    def loss_labels(self, outputs, targets, indices):
        src_logits = outputs['class_logits']  # [B, num_queries, num_classes+1]
        idx = self._get_src_permutation_idx(indices)
        
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                   dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight.to(src_logits.device))
        return {'loss_ce': loss_ce}
    
    def loss_boxes(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['bbox_pred'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / len(indices)
        
        loss_giou = 1 - torch.diag(self.generalized_box_iou(
            self.box_cxcywh_to_xyxy(src_boxes),
            self.box_cxcywh_to_xyxy(target_boxes)
        ))
        loss_giou = loss_giou.sum() / len(indices)
        
        return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}
    
    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_boxes(outputs, targets, indices))
        
        total_loss = sum(losses[k] * self.weight_dict[k] for k in losses.keys() if k in self.weight_dict)
        return total_loss, losses
    
    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    @staticmethod
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)
    
    @staticmethod
    def generalized_box_iou(boxes1, boxes2):
        assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
        
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - inter
        iou = inter / union
        
        lti = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rbi = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
        whi = (rbi - lti).clamp(min=0)
        areai = whi[:, :, 0] * whi[:, :, 1]
        
        return iou - (areai - union) / areai


# ============== 测试部分 ==============
if __name__ == '__main__':
    torch.manual_seed(42)
    
    # 模拟输出
    outputs = {
        'class_logits': torch.randn(2, 100, 81),  # B=2, num_queries=100, num_classes+1=81
        'bbox_pred': torch.rand(2, 100, 4)
    }
    
    # 模拟目标
    targets = []
    for _ in range(2):
        num_objs = 5
        targets.append({
            'labels': torch.randint(0, 80, (num_objs,)),  # 类别标签 (2, 5)
            'boxes': torch.rand(num_objs, 4)              # cx, cy, w, h
        })

    weight_dict = {'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}
    
    criterion = DETRLoss(num_classes=80, weight_dict=weight_dict)
    total_loss, losses = criterion(outputs, targets)
    
    print("Total Loss:", total_loss.item())
    for k, v in losses.items():
        print(f"{k}: {v.item():.4f}")
