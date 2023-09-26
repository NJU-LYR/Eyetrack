import torch.nn as nn
import torch


class BasicBlock(nn.Module):  # 定义ResNet网络18层和34层残差结构
    expansion = 1  # 残差结构当中卷积核的个数是否发生了变化

    def __init__(self, in_channel, out_channel, stride=1, downsample = None, **kwargs):
        # in_chanel 输入层深度 out_chanel 输出层深度 downsample 下采样参数，捷径中的卷积层
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False) # 使用batch normalization时不需要使用bias偏置
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x): # 输入矩阵的值
        identity = x     # 将输入矩阵输入到捷径中！
        if self.downsample is not None: # 捷径分支是否进行虚线操作，如果不等于None那么此时需要将输入经过下采样将矩阵的大小设置为相同的size
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # 此时通过第二层以后没有经过ReLU函数
        # 将捷径和卷积层的数据叠加以后才进行ReLU函数
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module): # 定义50层、101层、152层残差结构！
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4  # 定义了卷积核的变化 从64->256扩大了4倍，因此此处的expansion为4！！

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        # 第一层卷积层的stride为1 在实线和虚线残差结构中都是1因此此处将stride固定为1即可！
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        # 此时
        # 第二层卷积层的stride中实线stride是1，但是虚线stride为2 因此此时的stride是一个待传入的参数
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        # 第三层的输出的卷积核个数是卷积和格式的4倍，因此此时传入的卷积和个数就要增加为原来的expansion倍

        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion) #
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x): # 定义正向传播过程！
        identity = x
        if self.downsample is not None: # None对应实线残差结构 not None -> 虚线残差结构！
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


class ResNet(nn.Module):

    def __init__(self,
                 block, # block 用哪种残差结构：上面定义的两个BasicBlock对应18、32，Bottleneck对应50、101、152
                 blocks_num, # 所使用残差结构的数量 在论文中有对应的数量例如：50层（对应的列表为3、4、6、3）
                 num_classes=1000, # 训练集的分类个数
                 include_top=True, # 在ResNet上搭建更复杂的结构！
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  # 第一层的卷积层都是64所以此处指定定义为64即可

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, # ResNet网络第一层卷积层，3代表输入图像的Chanel 彩色图像是3！
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel) # 此处batch normalization的chanel就是第一层卷积层输出的chanel：in_chanel!
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])      # layer 定义每一个层的结构！
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1): # block：basicblock & bottleneck # channel: 残差结构中第一层的卷积核的个数
        # block_num对应了含有多少个残差结构！同时将步距stride默认设置为1！
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion: # 18层和34层的就会跳过这个步骤！
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = [] # 定义一个空列表
        layers.append(block(self.in_channel, # 输入卷积和的个数
                            channel, # 残差结构主分支第一个层的卷积核的个数
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):     # 正向传播过程
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

# 以上定义好ResNet网络的框架！

def resnet18(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)

def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
    # 34层的网络结构他的每一个层用到的数量分别是3，4，6，3 ！

def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)

