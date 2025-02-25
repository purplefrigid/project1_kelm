import torch  
import torch.nn as nn  
import torch.nn.functional as F  

# 定义残差块  
class ResidualBlock(nn.Module):  
    def __init__(self, in_channels, out_channels, stride=1):  
        super(ResidualBlock, self).__init__()  
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)  
        self.bn1 = nn.BatchNorm1d(out_channels)  
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)  
        self.bn2 = nn.BatchNorm1d(out_channels)  

        self.shortcut = nn.Sequential()  
        if stride != 1 or in_channels != out_channels:  
            self.shortcut = nn.Sequential(  
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),  
                nn.BatchNorm1d(out_channels)  
            )  

    def forward(self, x):  
        out = F.relu(self.bn1(self.conv1(x)))  
        out = self.bn2(self.conv2(out))  
        out += self.shortcut(x)  
        out = F.relu(out)  
        return out  

# 定义 ResNet 网络  
class ResNet(nn.Module):  
    def __init__(self, num_classes=97):  
        super(ResNet, self).__init__()  
        self.in_channels = 16  

        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)  
        self.bn1 = nn.BatchNorm1d(16)  
        self.layer1 = self._make_layer(16, 2, stride=1)  
        self.layer2 = self._make_layer(32, 2, stride=2)  
        self.layer3 = self._make_layer(64, 2, stride=2)  
        self.linear = nn.Linear(64 * 1, num_classes)  # 修改为 64 * 1  

    def _make_layer(self, out_channels, num_blocks, stride):  
        strides = [stride] + [1] * (num_blocks - 1)  
        layers = []  
        for stride in strides:  
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))  
            self.in_channels = out_channels  
        return nn.Sequential(*layers)  

    def forward(self, x):  
        out = F.relu(self.bn1(self.conv1(x)))  
        out = self.layer1(out)  
        out = self.layer2(out)  
        out = self.layer3(out)  
        out = F.avg_pool1d(out, kernel_size=3)  # 全局平均池化  
        out = out.view(out.size(0), -1)  # 展平  
        out = self.linear(out)  
        return out  

# 测试网络  
if __name__ == "__main__":  
    # 输入形状: (batchsize, 1, 12)  
    x = torch.randn(32, 1, 12)  
    model = ResNet()  
    output = model(x)  
    print(output.shape)  # 输出形状: (batchsize, 97)