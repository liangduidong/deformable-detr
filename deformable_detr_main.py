import torch
from utils.util import *
from Deformable_DETR_models.Deformable_DETR import DeformableDETR
from utils.loss import DETRLoss
from utils.dataset import create_data_loader

# ============== 完整训练示例 ==============

def main():
    """完整的训练流程示例"""
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    train_txt = 'data/train2017.txt' 
    val_txt = 'data/val2017.txt' 

    # 模型参数
    num_classes = 80  # COCO 数据集
    num_queries = 300
    
    # 初始化模型
    model = DeformableDETR(
        num_classes=num_classes,
        d_model=128,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_heads=8,
        d_ff=1024,
        num_levels=3,
        num_points=4,
        num_queries=num_queries,
        dropout=0.1
    ).to(device)
    
    # 计算参数总数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
  
    # 损失权重
    weight_dict = {
        'loss_ce': 1,      # 分类损失权重
        'loss_bbox': 5,    # L1 损失权重
        'loss_giou': 2,    # GIoU 损失权重
    }
    
    # 初始化损失函数
    criterion = DETRLoss(
        num_classes=num_classes,
        weight_dict=weight_dict,
        eos_coef=0.1
    ).to(device)
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # 学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    # 后处理器
    # postprocessor = PostProcess()
    
    # 构建数据加载器
    train_loader = create_data_loader(train_txt, batch_size=2, shuffle=True, num_workers=0)
    # val_loader = create_data_loader(val_txt, batch_size=2, shuffle=False, num_workers=0)

    # 训练
    num_epochs = 300
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch+1}')
        # 训练（需要实际的 data_loader）
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, device)
        
        # 评估（需要实际的 data_loader）
        # val_loss, predictions, targets = evaluate(model, criterion, postprocessor, val_loader, device)
        
        lr_scheduler.step()
        
        # 保存检查点
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            }, f'checkpoint/checkpoint_epoch_{epoch+1}.pth')
    
    print("训练完成！")


# ============== 7. 测试代码 ==============

if __name__ == "__main__":
    main()
    
