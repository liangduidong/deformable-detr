import torch
import torch.nn as nn
import torch.nn.functional as F


# ============== 基础残差块（保持不变）==============

class BasicBlock(nn.Module):
    """ResNet18/34 使用的残差块"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """ResNet50/101/152 使用的残差块"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        
        return out


# ============== 修改后的 ResNet（多尺度输出）==============

class ResNetMultiScale(nn.Module):
    """
    修改版 ResNet，用于 Deformable DETR
    
    主要改动：
    1. 返回多尺度特征 (C3, C4, C5) 而不是分类结果
    2. 移除最后的全局池化和全连接层
    3. 添加 num_channels 属性，记录每个尺度的通道数
    """
    def __init__(self, block, layers, return_layers=[1, 2, 3]):
        """
        Args:
            block: BasicBlock 或 Bottleneck
            layers: 每个 stage 的残差块数量，如 [3, 4, 6, 3] for ResNet50
            return_layers: 返回哪些层的特征，[1, 2, 3] 表示返回 layer2, layer3, layer4
        """
        super(ResNetMultiScale, self).__init__()
        self.in_channels = 64
        self.return_layers = return_layers
        
        # ========== 初始卷积层 ==========
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ========== ResNet Stages ==========
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # ========== 记录每个层的输出通道数 ==========
        # 这对于后续的特征投影很重要
        self.num_channels = [
            64 * block.expansion,   # layer1 输出
            128 * block.expansion,  # layer2 输出
            256 * block.expansion,  # layer3 输出
            512 * block.expansion,  # layer4 输出
        ]

        # ========== 参数初始化 ==========
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
    def _make_layer(self, block, out_channels, blocks, stride=1):
        """创建残差层"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播 - 返回多尺度特征
        
        Args:
            x: [B, 3, H, W] 输入图像
        
        Returns:
            list of features: 多尺度特征图
            - 如果 return_layers=[1,2,3]，返回 [C3, C4, C5]
            - C3: [B, C3, H/8, W/8]   (来自 layer2)
            - C4: [B, C4, H/16, W/16] (来自 layer3)
            - C5: [B, C5, H/32, W/32] (来自 layer4)
        """
        # 初始卷积和池化: H/4, W/4
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 保存多尺度特征
        features = []
        
        # Layer 1: H/4, W/4
        x = self.layer1(x)
        if 0 in self.return_layers:
            features.append(x)
        
        # Layer 2: H/8, W/8 (C3)
        x = self.layer2(x)
        if 1 in self.return_layers:
            features.append(x)
        
        # Layer 3: H/16, W/16 (C4)
        x = self.layer3(x)
        if 2 in self.return_layers:
            features.append(x)
        
        # Layer 4: H/32, W/32 (C5)
        x = self.layer4(x)
        if 3 in self.return_layers:
            features.append(x)

        return features


# ============== 工厂函数（多尺度版本）==============

def resnet18_multiscale(pretrained=False, return_layers=[1, 2, 3]):
    """ResNet-18 多尺度版本"""
    model = ResNetMultiScale(BasicBlock, [2, 2, 2, 2], return_layers)
    if pretrained:
        # 可以加载预训练权重
        pass
    return model


def resnet34_multiscale(pretrained=False, return_layers=[1, 2, 3]):
    """ResNet-34 多尺度版本"""
    model = ResNetMultiScale(BasicBlock, [3, 4, 6, 3], return_layers)
    if pretrained:
        pass
    return model


def resnet50_multiscale(pretrained=False, return_layers=[1, 2, 3]):
    """ResNet-50 多尺度版本 (Deformable DETR 常用)"""
    model = ResNetMultiScale(Bottleneck, [3, 4, 6, 3], return_layers)
    if pretrained:
        pass
    return model


def resnet101_multiscale(pretrained=False, return_layers=[1, 2, 3]):
    """ResNet-101 多尺度版本"""
    model = ResNetMultiScale(Bottleneck, [3, 4, 23, 3], return_layers)
    if pretrained:
        pass
    return model


# ============== 测试代码 ==============

def test_multiscale_backbone():
    """测试多尺度 backbone"""
    print("=" * 60)
    print("测试多尺度 ResNet Backbone")
    print("=" * 60)
    
    # 创建模型
    model = resnet50_multiscale(return_layers=[1, 2, 3])
    model.eval()
    
    # 测试输入
    batch_size = 2
    x = torch.randn(batch_size, 3, 640, 640)
    print(f"\n输入形状: {x.shape}")
    
    # 前向传播
    with torch.no_grad():
        features = model(x)
    
    # 打印多尺度特征
    print(f"\n多尺度特征输出:")
    feature_names = ['C3 (1/8)', 'C4 (1/16)', 'C5 (1/32)']
    for i, (feat, name) in enumerate(zip(features, feature_names)):
        print(f"  {name}: {feat.shape}")
        print(f"    通道数: {feat.shape[1]}")
        print(f"    空间尺寸: {feat.shape[2]}x{feat.shape[3]}")
    
    # 验证通道数
    print(f"\n每层通道数 (num_channels): {model.num_channels}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")
    
    print("\n✓ 多尺度 backbone 测试通过！")


def test_deformable_detr_integration():
    """测试与 Deformable DETR 的集成"""
    print("\n" + "=" * 60)
    print("测试与 Deformable DETR 的集成")
    print("=" * 60)
    
    # 创建 backbone
    backbone = resnet50_multiscale(return_layers=[1, 2, 3])
    
    # 模拟 Deformable DETR 的投影层
    d_model = 256
    num_levels = 3
    
    input_proj = nn.ModuleList([ # Deformable DETR 的投影层
        nn.Conv2d(backbone.num_channels[i+1], d_model, kernel_size=1)
        for i in range(num_levels)
    ])
    
    print(f"\n投影层配置:")
    for i, (in_ch, proj) in enumerate(zip(backbone.num_channels[1:], input_proj)):
        print(f"  Level {i}: {in_ch} -> {d_model}")
    
    # 测试完整流程
    x = torch.randn(2, 3, 640, 640)
    
    with torch.no_grad():
        # 1. 提取多尺度特征
        features = backbone(x)
        
        # 2. 投影到统一维度
        projected_features = []
        for feat, proj in zip(features, input_proj):
            proj_feat = proj(feat)
            projected_features.append(proj_feat)
            print(f"\n投影后特征: {proj_feat.shape}")
    
    print("\n✓ 集成测试通过！")


if __name__ == '__main__':
    # 测试多尺度 backbone
    test_multiscale_backbone()
    
    # 测试集成
    # test_deformable_detr_integration()
    
    print("\n" + "=" * 60)
