import torch.nn as nn
import torch

# 定义贝叶斯神经网络模型  
class BayesianNN(nn.Module):  
    def __init__(self):  
        super(BayesianNN, self).__init__()  
        # 两层卷积层  
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)  
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)  
        
        # 四层线性层  
        self.fc1 = nn.Linear(32 * 5 * 5, 128)  # 假设输入经过卷积后展平为 32 * 5 * 5  
        self.fc2 = nn.Linear(128, 64)  
        self.fc3 = nn.Linear(64, 32)  
        self.fc4 = nn.Linear(32, 7)  # 输出 7 个参数  

        # 激活函数  
        self.relu = nn.ReLU()  
        self.flatten = nn.Flatten()  

    def forward(self, x):  
        # 卷积层  
        x = self.relu(self.conv1(x))  
        x = self.relu(self.conv2(x))  
        
        # 展平  
        x = self.flatten(x)  
        
        # 全连接层  
        x = self.relu(self.fc1(x))  
        x = self.relu(self.fc2(x))  
        x = self.relu(self.fc3(x))  
        x = self.fc4(x)  # 输出层  
        return x  