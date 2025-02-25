import torch.nn as nn
import torch
# def init_weight(m):
#     if type(m) == nn.Linear or type(m) == nn.Conv1d:
#         nn.init.kaiming_uniform_()


class F_Net_1D(nn.Module):
        # tensor(batchsize, channel, Length)
        # Conv2d   tensor(batchsize, channel, H,W)
    def __init__(self):
        super(F_Net_1D, self).__init__()
        self.conv1=nn.Conv1d(in_channels=1, out_channels=32, stride=1, padding=1, kernel_size=3)
        self.conv2=nn.Conv1d(in_channels=32, out_channels=64, stride=1, padding=1, kernel_size=3)
        self.conv3=nn.Conv1d(in_channels=64, out_channels=128, stride=1, padding=1, kernel_size=3)
        self.conv4=nn.Conv1d(in_channels=128, out_channels=256, stride=1, padding=1, kernel_size=3)
        self.fc1=nn.Linear(in_features=1792, out_features=896)
        self.fc2=nn.Linear(in_features=896, out_features=772)
        self.fc3=nn.Linear(in_features=772, out_features=448)
        self.fc4=nn.Linear(in_features=448, out_features=336)
        self.fc5=nn.Linear(in_features=336, out_features=224)
        self.fc6=nn.Linear(in_features=224, out_features=161)
        self.fc7=nn.Linear(in_features=161, out_features=97)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1, ceil_mode=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x=self.relu(self.conv2(x))
        x=self.relu(self.conv3(x))
        x=self.relu(self.conv4(x))
        x=self.maxpool(x)
        x = x.view(x.size(0), -1)
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.relu(self.fc3(x))
        x=self.relu(self.fc4(x))
        x=self.relu(self.fc5(x))
        x=self.relu(self.fc6(x))
        x=self.relu(self.fc7(x))
        return x  #tensor(bs,351)

class F_Net_2D(nn.Module):

    # tensor(batchsize, channel, Length)
    # Conv2d   tensor(batchsize, channel, H,W)
    def __init__(self):
        super(F_Net_2D, self).__init__()
        # 卷积向下取整，池化向上取整
        self.CRM = nn.Sequential(  # (bs,1,8)==》(bs,32,8)
            nn.Conv1d(in_channels=1, out_channels=32, stride=1, padding=1, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=32, out_channels=64, stride=1, padding=1, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=64, out_channels=128, stride=1, padding=1, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=256, stride=1, padding=1, kernel_size=3),
            nn.ReLU(inplace=True),            
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1, ceil_mode=False),  # bs,512,3
        )

        self.FCL = nn.Sequential(
            nn.Linear(in_features=1792, out_features=896),
            # nn.Dropout(0.1),  # Dropout 层，丢弃 50%
            nn.ReLU(inplace=True),
            nn.Linear(in_features=896, out_features=772),
            # nn.Dropout(0.2),  # Dropout 层，丢弃 50%
            nn.ReLU(inplace=True),
            nn.Linear(in_features=772, out_features=448),
            # nn.Dropout(0.3),  # Dropout 层，丢弃 50%
            nn.ReLU(inplace=True),
            nn.Linear(in_features=448, out_features=336),
            # nn.Dropout(0.4),  # Dropout 层，丢弃 50%
            nn.ReLU(inplace=True),
            nn.Linear(in_features=336, out_features=224),
            # nn.Dropout(0.5),  # Dropout 层，丢弃 50%
            nn.ReLU(inplace=True),
            nn.Linear(in_features=224, out_features=161),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=161, out_features=97),
        )

    def forward(self, x):
        x = self.CRM(x)
        x = x.view(x.size(0), -1)
        y_pred = self.FCL(x)
        return y_pred  # tensor(bs,351)

class F_Net_3D(nn.Module):

    # tensor(batchsize, channel, Length)
    # Conv2d   tensor(batchsize, channel, H,W)
    def __init__(self):
        super(F_Net_3D, self).__init__()
        # 卷积向下取整，池化向上取整
        self.CRM = nn.Sequential(  # (bs,1,8)==》(bs,32,8)
            nn.Conv1d(in_channels=1, out_channels=32, stride=1, padding=1, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=32, out_channels=64, stride=1, padding=1, kernel_size=3),
            nn.ReLU(inplace=True),       
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1, ceil_mode=False),  # bs,512,3
        )

        self.FCL = nn.Sequential(
            nn.Linear(in_features=448, out_features=336),
            # nn.Dropout(0.4),  # Dropout 层，丢弃 50%
            nn.ReLU(inplace=True),
            nn.Linear(in_features=336, out_features=224),
            # nn.Dropout(0.5),  # Dropout 层，丢弃 50%
            nn.ReLU(inplace=True),
            nn.Linear(in_features=224, out_features=161),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=161, out_features=97),
        )

    def forward(self, x):
        x = self.CRM(x)
        x = x.view(x.size(0), -1)
        y_pred = self.FCL(x)
        return y_pred # tensor(bs,351)
"""
    def __init__(self):
        super(MH_Net, self).__init__()
        self.model = torch.nn.Sequential(
            # input:1*321*9; output:8*321*9
            torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),    # output:8*160*4

            # input:8*160*4; output:16*160*4
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),    # output:16*80*2

            # input:16*80*2; output:32*80*2
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),

            torch.nn.Flatten(),
            torch.nn.Linear(in_features=32 * 80 * 2, out_features=2560),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2560, out_features=1280),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=1280, out_features=321),
        )

    def forward(self, input):
        y_pred = self.model(input)
        return y_pred
"""





"""
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),F_Net_1D
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

"""
if __name__ == '__main__':
    # model = torchvision.models.AlexNet()
    model = F_Net_3D()
    input = torch.ones((30,1,12))
    output = model(input)
    print(output.shape)

    # input = torch.randn(8, 3, 224, 224)
    # # torch.randn[batch, channel, height, width]，表示batch_size=8， 3通道（灰度图像为1），图片尺寸：224x224
    # out = model(input)
    # print(out.shape)
