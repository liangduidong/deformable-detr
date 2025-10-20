import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import os


class COCODataset(Dataset):
    """
    COCO2017 数据集加载器
    
    数据格式：
    每行: 图像路径 x1,y1,x2,y2,label x1,y1,x2,y2,label ...
    例如: path/to/img.jpg 217,240,256,298,39 1,240,347,427,60 ...
    """
    def __init__(self, txt_file, transform=None, target_size=(800, 800)):
        """
        Args:
            txt_file: 标注文件路径
            transform: 图像变换
            target_size: 目标图像尺寸 (H, W)
        """
        self.txt_file = txt_file
        self.target_size = target_size
        
        # 读取数据
        self.data = []
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(line)
        
        # 图像变换
        if transform is None:
            self.transform = T.Compose([
                T.Resize(target_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        返回:
            image: tensor [3, H, W]
            target: dict {
                'labels': tensor [num_objects],
                'boxes': tensor [num_objects, 4],  # 归一化的 [cx, cy, w, h]
                'orig_size': tensor [2],  # 原始图像尺寸 [H, W]
                'size': tensor [2]  # 调整后的图像尺寸 [H, W]
            }
        """
        line = self.data[idx]
        parts = line.split()
        
        # 解析图像路径
        img_path = parts[0]
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"错误: 无法加载图像 {img_path}: {e}")
            # 返回一个空白图像
            image = Image.new('RGB', (640, 480), color=(0, 0, 0))
        
        orig_w, orig_h = image.size
        
        # 解析标注框
        labels = []
        boxes = []
        
        for i in range(1, len(parts)):
            bbox_info = parts[i].split(',')
            if len(bbox_info) == 5:
                x1, y1, x2, y2, label = map(int, bbox_info)
                
                # 跳过无效框
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # 转换为中心坐标格式 [cx, cy, w, h] 并归一化到 [0, 1]
                cx = (x1 + x2) / 2.0 / orig_w
                cy = (y1 + y2) / 2.0 / orig_h
                w = (x2 - x1) / orig_w
                h = (y2 - y1) / orig_h
                
                # 确保在 [0, 1] 范围内
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))
                
                boxes.append([cx, cy, w, h])
                labels.append(label)
        
        # 应用图像变换
        image = self.transform(image)
        
        # 转换为 tensor
        if len(boxes) == 0:
            # 如果没有标注框，创建一个虚拟框（背景）
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {
            "labels": labels,
            "boxes": boxes,
            "orig_size": torch.tensor([orig_h, orig_w]),  # 原始尺寸
            "size": torch.tensor(self.target_size)  # 调整后的尺寸
        }
        
        return image, target
    
    @staticmethod
    def collate_fn(batch):
        """
        自定义 collate 函数
        
        Args:
            batch: list of (image, target) tuples
        
        Returns:
            images: tensor [B, 3, H, W]
            targets: list of dict
        """
        images = torch.stack([item[0] for item in batch], dim=0)
        targets = [item[1] for item in batch]
        return images, targets


# ============== 数据加载示例 ==============

def create_data_loader(txt_file, batch_size=2, shuffle=True, num_workers=0):
    """
    创建数据加载器
    
    Args:
        txt_file: 标注文件路径
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
    
    Returns:
        DataLoader
    """
    dataset = COCODataset(txt_file)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True
    )
    
    return dataloader


# ============== 测试代码 ==============

def test_dataset(txt_file):
    """测试数据集加载"""
    print("=" * 60)
    print("测试 COCO 数据集加载")
    print("=" * 60)
    
    # 创建数据集
    dataset = COCODataset(txt_file, target_size=(800, 800))
    print(f"\n数据集大小: {len(dataset)}")
    
    # 测试单个样本
    if len(dataset) > 0:
        image, target = dataset[0]
        print(f"\n第一个样本:")
        print(f"  图像形状: {image.shape}")
        print(f"  标签数量: {len(target['labels'])}")
        print(f"  标签: {target['labels']}")
        print(f"  边界框形状: {target['boxes'].shape}")
        print(f"  边界框范围: [{target['boxes'].min():.3f}, {target['boxes'].max():.3f}]")
        print(f"  原始尺寸: {target['orig_size']}")
        print(f"  调整后尺寸: {target['size']}")
        
        # 显示前 3 个框
        if len(target['boxes']) > 0:
            print(f"\n前 3 个边界框 (归一化 cx, cy, w, h):")
            for i in range(min(3, len(target['boxes']))):
                box = target['boxes'][i]
                label = target['labels'][i]
                print(f"    框 {i+1}: [{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f}], 类别: {label}")
    
    # 测试 DataLoader
    print(f"\n" + "=" * 60)
    print("测试 DataLoader")
    print("=" * 60)
    
    dataloader = create_data_loader(txt_file, batch_size=2, shuffle=False)
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  图像形状: {images.shape}")
        print(f"  Batch 大小: {len(targets)}")
        
        for i, target in enumerate(targets):
            print(f"  样本 {i+1}:")
            print(f"    对象数量: {len(target['labels'])}")
            print(f"    标签: {target['labels'].tolist()}")
        
        # 只测试第一个 batch
        if batch_idx == 0:
            break
    
    print("\n✓ 数据集测试完成！")


# ============== 数据统计工具 ==============

def analyze_dataset(txt_file):
    """分析数据集统计信息"""
    print("=" * 60)
    print("数据集统计分析")
    print("=" * 60)
    
    dataset = COCODataset(txt_file)
    
    total_objects = 0
    class_counts = {}
    box_sizes = []
    
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        total_objects += len(target['labels'])
        
        for label in target['labels'].tolist():
            class_counts[label] = class_counts.get(label, 0) + 1
        
        for box in target['boxes']:
            # 计算框的面积
            area = box[2] * box[3]  # w * h
            box_sizes.append(area)
    
    print(f"\n总图像数: {len(dataset)}")
    print(f"总对象数: {total_objects}")
    print(f"平均每张图像对象数: {total_objects / len(dataset):.2f}")
    
    print(f"\n类别分布 (前 10 个):")
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    for cls, count in sorted_classes[:10]:
        print(f"  类别 {cls}: {count} 个对象")
    
    if box_sizes:
        import numpy as np
        box_sizes = np.array(box_sizes)
        print(f"\n边界框大小统计:")
        print(f"  最小面积: {box_sizes.min():.4f}")
        print(f"  最大面积: {box_sizes.max():.4f}")
        print(f"  平均面积: {box_sizes.mean():.4f}")
        print(f"  中位数面积: {np.median(box_sizes):.4f}")


# ============== 可视化工具 ==============

def visualize_sample(dataset, idx=0, save_path=None):
    """可视化数据集样本"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    image, target = dataset[idx]
    
    # 反归一化图像
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype('uint8')
    
    # 创建图形
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    # 绘制边界框
    h, w = image.shape[:2]
    for box, label in zip(target['boxes'], target['labels']):
        cx, cy, bw, bh = box.tolist()
        
        # 转换为像素坐标
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        width = bw * w
        height = bh * h
        
        # 绘制矩形
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        
        # 添加标签
        ax.text(x1, y1 - 5, f'Class {label.item()}',
                bbox=dict(facecolor='yellow', alpha=0.5),
                fontsize=10, color='black')
    
    ax.axis('off')
    ax.set_title(f'Sample {idx} - {len(target["labels"])} objects')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"可视化结果已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============== 主函数 ==============

if __name__ == "__main__":
    # 设置您的数据文件路径
    txt_file = "path/to/your/annotations.txt"
    
    # 如果文件不存在，创建示例数据
    if not os.path.exists(txt_file):
        print("创建示例数据文件...")
        sample_data = """E:/learnt/postgraduate/Dataset/COCO2017/val2017/000000397133.jpg 217,240,256,298,39 1,240,347,427,60 388,69,498,347,0
E:/learnt/postgraduate/Dataset/COCO2017/val2017/000000037777.jpg 102,118,110,135,58 26,215,88,229,56
E:/learnt/postgraduate/Dataset/COCO2017/val2017/000000252219.jpg 326,174,397,371,0 9,167,131,393,0"""
        
        with open('sample_annotations.txt', 'w', encoding='utf-8') as f:
            f.write(sample_data)
        
        txt_file = 'sample_annotations.txt'
        print(f"已创建示例文件: {txt_file}")
    
    # 测试数据集
    print("\n" + "=" * 60)
    print("开始测试...")
    print("=" * 60)
    
    try:
        test_dataset(txt_file)
        print("\n")
        analyze_dataset(txt_file)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("使用说明:")
    print("=" * 60)
    print("""
1. 基本使用:
   dataset = COCODataset('your_annotations.txt')
   dataloader = create_data_loader('your_annotations.txt', batch_size=4)

2. 带数据增强:
   train_dataset = COCODatasetWithAugmentation('train.txt', is_train=True)
   val_dataset = COCODatasetWithAugmentation('val.txt', is_train=False)

3. 使用 DataLoader:
   for images, targets in dataloader:
       outputs = model(images)
       loss = criterion(outputs, targets)

4. 可视化:
   dataset = COCODataset('your_annotations.txt')
   visualize_sample(dataset, idx=0, save_path='sample.png')
    """)