import torch
import torch.nn as nn
import torch.nn.functional as F

# ============== 后处理 ==============

class PostProcess(nn.Module):
    """后处理模块：将模型输出转换为可用的检测结果"""
    
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """
        outputs: dict with 'class_logits' and 'bbox_pred'
        target_sizes: tensor of shape [batch_size, 2] containing target h, w
        
        Returns:
            list of dict, each containing:
                - scores: [num_keep]
                - labels: [num_keep]
                - boxes: [num_keep, 4] in [x0, y0, x1, y1] format
        """
        out_logits = outputs['class_logits']
        out_bbox = outputs['bbox_pred']
        
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        
        # 获取类别概率和标签
        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)  # 排除背景类
        
        # 将框从 [0, 1] 转换为图像尺寸
        boxes = self.box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        
        results = []
        for s, l, b in zip(scores, labels, boxes):
            results.append({'scores': s, 'labels': l, 'boxes': b})
        
        return results
    
    @staticmethod
    def box_cxcywh_to_xyxy(x):
        """中心格式转换为角点格式"""
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

# ============== 训练函数 ==============

def train_one_epoch(model, criterion, data_loader, optimizer, device):
    """训练一个 epoch"""
    model.train()
    criterion.train()
    
    total_loss = 0
    total_loss_ce = 0
    total_loss_bbox = 0
    total_loss_giou = 0
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        loss, loss_dict = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_loss_ce += loss_dict['loss_ce'].item()
        total_loss_bbox += loss_dict['loss_bbox'].item()
        total_loss_giou += loss_dict['loss_giou'].item()
        
        # 打印训练进度
        if (batch_idx+1) % 100 == 0:
            print(f'Batch: {(batch_idx+1)}/{len(data_loader)}, Loss: {total_loss / (batch_idx+1):.4f}')
            print(f"Loss_ce: {total_loss_ce / (batch_idx+1):.4f}")
            print(f"Loss_bbox: {total_loss_bbox / (batch_idx+1):.4f}")
            print(f"Loss_giou: {total_loss_giou / (batch_idx+1):.4f}\n")
        
    return total_loss / len(data_loader)


# ============== 评估函数 ==============
@torch.no_grad()
def evaluate(model, criterion, postprocessor, data_loader, device):
    """评估模型"""
    model.eval()
    criterion.eval()
    
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    for images, targets in data_loader:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        loss, loss_dict = criterion(outputs, targets)
        total_loss += loss.item()
        
        # 后处理
        target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        results = postprocessor(outputs, target_sizes)
        
        all_predictions.extend(results)
        all_targets.extend(targets)
    
    return total_loss / len(data_loader), all_predictions, all_targets

# ============== 推理函数 ==============
@torch.no_grad()
def inference(model, image, device, conf_threshold=0.7):
    """
    单张图像推理
    
    Args:
        model: DETR 模型
        image: PIL Image or tensor [C, H, W]
        device: 设备
        conf_threshold: 置信度阈值
    
    Returns:
        dict with 'scores', 'labels', 'boxes'
    """
    model.eval()
    
    # 预处理
    if not isinstance(image, torch.Tensor):
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = transform(image)
    
    image = image.unsqueeze(0).to(device)  # [1, C, H, W]
    
    # 前向传播
    outputs = model(image)
    
    # 后处理
    postprocessor = PostProcess()
    target_sizes = torch.tensor([[image.shape[2], image.shape[3]]]).to(device)
    results = postprocessor(outputs, target_sizes)[0]
    
    # 过滤低置信度检测
    keep = results['scores'] > conf_threshold
    results = {
        'scores': results['scores'][keep],
        'labels': results['labels'][keep],
        'boxes': results['boxes'][keep]
    }
    
    return results


if __name__ == '__main__':
    print('测试中...')
