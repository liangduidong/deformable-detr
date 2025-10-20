import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module): # 定义一个BasicBlock类.适用resnet18、34的残差结构
    expansion = 1 # 定义一个类变量expansion，表示输出通道数与输入通道数的扩展倍数

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__() # 调用父类的构造函数，初始化BasicBlock类的实例
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False) # 定义第一个卷积层
        self.bn1 = nn.BatchNorm2d(out_channels) # 定义第一个批归一化层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False) # 定义第二个卷积层
        self.bn2 = nn.BatchNorm2d(out_channels) # 定义第二个批归一化层
        self.downsample = downsample # 定义下采样层
        self.stride = stride # 定义步幅
        self.relu = nn.ReLU(inplace=True) # 定义ReLU激活函数
    
    def forward(self, x): # 定义前向传播函数
        identity = x # 保存输入张量作为残差连接
        if self.downsample is not None: # 如果存在下采样层
            identity = self.downsample(x) # 对输入张量进行下采样

        out = self.conv1(x) # 通过第一个卷积层
        out = self.bn1(out) # 通过第一个批归一化层
        out = self.relu(out) # 通过ReLU激活函数

        out = self.conv2(out) # 通过第二个卷积层
        out = self.bn2(out) # 通过第二个批归一化层

        out += identity # 将残差连接加到输出上
        out = self.relu(out) # 通过ReLU激活函数

        return out # 返回输出张量

class Bottleneck(nn.Module): # 定义一个Bottleneck类.适用resnet50、101、152的残差结构
    expansion = 4 # 定义一个类变量expansion，表示输出通道数与输入通道数的扩展倍数

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__() # 调用父类的构造函数，初始化Bottleneck类的实例
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False) # 定义第一个卷积层
        self.bn1 = nn.BatchNorm2d(out_channels) # 定义第一个批归一化层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False) # 定义第二个卷积层
        self.bn2 = nn.BatchNorm2d(out_channels) # 定义第二个批归一化层
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False) # 定义第三个卷积层
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion) # 定义第三个批归一化层
        self.downsample = downsample # 定义下采样层
        self.stride = stride # 定义步幅
        self.relu = nn.ReLU(inplace=True) # 定义ReLU激活函数 
    
    def forward(self, x): # 定义前向传播函数
        identity = x # 保存输入张量作为残差连接
        if self.downsample is not None: # 如果存在下采样层
            identity = self.downsample(x) # 对输入张量进行下采样

        out = self.conv1(x) # 通过第一个卷积层
        out = self.bn1(out) # 通过第一个批归一化层
        out = self.relu(out) # 通过ReLU激活函数

        out = self.conv2(out) # 通过第二个卷积层
        out = self.bn2(out) # 通过第二个批归一化层
        out = self.relu(out) # 通过ReLU激活函数

        out = self.conv3(out) # 通过第三个卷积层
        out = self.bn3(out) # 通过第三个批归一化层

        out += identity # 将残差连接加到输出上
        out = self.relu(out) # 通过ReLU激活函数
        return out # 返回输出张量
    
class ResNet(nn.Module): # 定义一个ResNet类，表示ResNet网络
    def __init__(self, block, layers, num_classes=1000, out_channels=256):
        super(ResNet, self).__init__() # 调用父类的构造函数，初始化ResNet类的实例
        self.in_channels = 64 # 定义初始输入通道数为64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False) # 定义第一个卷积层
        self.bn1 = nn.BatchNorm2d(self.in_channels) # 定义第一个批归一化层
        self.relu = nn.ReLU(inplace=True) # 定义ReLU激活函数
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 定义最大池化层

        self.layer1 = self._make_layer(block, 64, layers[0]) # 定义第一个残差层
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 定义第二个残差层
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # 定义第三个残差层
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # 定义第四个残差层

        # 将输出通道压缩到256（与Transformer的d_model匹配）
        self.input_proj = nn.Conv2d(2048, out_channels, kernel_size=1)

        for m in self.modules(): # 初始化网络参数
            if isinstance(m, nn.Conv2d): # 如果是卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # 使用Kaiming正态初始化
            elif isinstance(m, nn.BatchNorm2d): # 如果是批归一化层
                nn.init.constant_(m.weight, 1) # 将权重初始化为1
                nn.init.constant_(m.bias, 0) # 将偏置初始化为0
 
    def _make_layer(self, block, out_channels, blocks, stride=1): # 定义一个辅助函数，用于创建残差层
        downsample = None # 初始化下采样层为None
        if stride != 1 or self.in_channels != out_channels * block.expansion: # 如果步幅不为1或输入通道数不等于输出通道数乘以扩展倍数
            downsample = nn.Sequential( # 定义下采样层
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False), # 1x1卷积层
                nn.BatchNorm2d(out_channels * block.expansion), # 批归一化层
            ) 

        layers = [] # 初始化一个空列表，用于存储残差块
        layers.append(block(self.in_channels, out_channels, stride, downsample)) # 添加第一个残差块，可能包含下采样
        self.in_channels = out_channels * block.expansion # 更新输入通道数

        for _ in range(1, blocks): # 添加剩余的残差块
            layers.append(block(self.in_channels, out_channels)) # 添加残差块

        return nn.Sequential(*layers) # 返回一个包含所有残差块的序列容器
    
    def forward(self, x): # 定义前向传播函数
        x = self.conv1(x) # 通过第一个卷积层
        x = self.bn1(x) # 通过第一个批归一化层
        x = self.relu(x) # 通过ReLU激活函数
        x = self.maxpool(x) # 通过最大池化层

        x = self.layer1(x) # 通过第一个残差层
        x = self.layer2(x) # 通过第二个残差层
        x = self.layer3(x) # 通过第三个残差层
        x = self.layer4(x) # 通过第四个残差层

        x = self.input_proj(x) # 通过输入投影

        return x # 返回输出张量
def resnet18(num_classes=1000, out_channels=256): # 定义一个辅助函数，创建ResNet-18模型
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, out_channels)
def resnet34(num_classes=1000, out_channels=256): # 定义一个辅助函数，创建ResNet-34模型
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, out_channels)
def resnet50(num_classes=1000, out_channels=256): # 定义一个辅助函数，创建Res
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, out_channels)
def resnet101(num_classes=1000, out_channels=256): # 定义一个辅助函数，创建ResNet-101模型
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, out_channels)
def resnet152(num_classes=1000, out_channels=256): # 定义一个辅助函数，创建ResNet-152模型
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, out_channels)


def test():
    model = resnet18(num_classes=10) # 创建一个ResNet-18模型，输出类别数为10
    x = torch.randn(1, 3, 224, 224) # 创建一个随机输入张量，形状为(1, 3, 224, 224)
    y = model(x) # 将输入张量传递给模型，得到输出张量
    print(y.size()) # 打印输出张量的形状

if __name__ == '__main__':
    test() # 调用测试函数Net-50模型