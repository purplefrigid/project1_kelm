# """
# In this file the basic ModAL PyTorch DeepActiveLearner workflow is explained 
# through an example on the MNIST dataset and the MC-Dropout-Bald query strategy.
# """
# import numpy as np
# import torch
# # import of query strategies
# from modAL.dropout import mc_dropout_bald
# from modAL.models import DeepActiveLearner
# from skorch import NeuralNetClassifier
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision.datasets import MNIST
# from torchvision.transforms import ToTensor
# from sklearn.metrics import accuracy_score  

# # Standard Pytorch Model (Visit the PyTorch documentation for more details)
# class Torch_Model(nn.Module):
#     def __init__(self,):
#         super(Torch_Model, self).__init__()
#         self.convs = nn.Sequential(
#             nn.Conv2d(1, 32, 3),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 3),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Dropout(0.25)
#         )
#         self.fcs = nn.Sequential(
#             nn.Linear(12*12*64, 128),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128, 10),
#         )

#     def forward(self, x):
#         out = x
#         out = self.convs(out)
#         out = out.view(-1, 12*12*64)
#         out = self.fcs(out)
#         return out


# torch_model = Torch_Model()
# """
# You can acquire from the layer_list the dropout_layer_indexes, which can then be passed on 
# to the query strategies to decide which dropout layers should be active for the predictions. 
# When no dropout_layer_indexes are passed, all dropout layers will be activated on default. 
# """
# layer_list = list(torch_model.modules())

# device = "cuda" if torch.cuda.is_available() else "cpu"
# # 加载模型  
# net_loaded =  NeuralNetClassifier(Torch_Model,
#                                  criterion=torch.nn.CrossEntropyLoss,
#                                  optimizer=torch.optim.Adam,
#                                  train_split=None,
#                                  verbose=1,
#                                  device=device)
 

# # 初始化模型后加载参数  
# net_loaded.initialize()  
# net_loaded.load_params(f_params='model_params.pt')
# # Load the Dataset
# mnist_data = MNIST('.', download=True, transform=ToTensor())
# dataloader = DataLoader(mnist_data, shuffle=True, batch_size=60000)
# X, y = next(iter(dataloader))

# # read training data
# X_train, X_test, y_train, y_test = X[:50000], X[50000:], y[:50000], y[50000:]
# X_train = X_train.reshape(50000, 1, 28, 28)
# X_test = X_test.reshape(10000, 1, 28, 28)

# # for i in range(500):
# #     b=y_test[i]
# #     a= X_test[i]
# #     y_pred = learner.predict(X_test[i])
# # 使用加载的模型进行预测  
# y_pred = net_loaded.predict(X_test)  

# # 如果需要预测概率分布  
# # y_proba = net_loaded.predict_proba(X_test)  

# print("Predicted labels:", y_pred)  
# print("Real labels::", y_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))  
####################################################################################################################################
# import torch  
# from torch import nn  
# from skorch import NeuralNetClassifier  
# from sklearn.datasets import make_classification  
# from sklearn.model_selection import train_test_split  
# from sklearn.metrics import accuracy_score  

# # 定义一个简单的 PyTorch 模型  
# class ClassifierModule(nn.Module):  
#     def __init__(self, num_features, num_classes):  
#         super().__init__()  
#         self.fc = nn.Linear(num_features, num_classes)  

#     def forward(self, X):  
#         return self.fc(X)  

# # 生成一个简单的分类数据集  
# X, y = make_classification(1000, 20, n_classes=2, random_state=42)  
# X = X.astype('float32')  # skorch 需要 float32 类型  
# y = y.astype('int64')    # 标签需要是 int64 类型  
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# # 使用 NeuralNetClassifier 封装模型  
# net = NeuralNetClassifier(  
#     ClassifierModule,  
#     module__num_features=20,  # 传递给模型的参数  
#     module__num_classes=2,  
#     max_epochs=10,  
#     lr=0.1,  
#     iterator_train__shuffle=True,  # 打乱训练数据  
# )  

# # 训练模型  
# net.fit(X_train, y_train)  

# # 预测  
# y_pred = net.predict(X_test)  

# # 评估  
# print("Accuracy:", accuracy_score(y_test, y_pred))
#################################################################################################################################################
# import pandas as pd  

# # 输入和输出文件路径  
# input_file = "input.xlsx"  # 原始文件名  
# output_file = "output.xlsx"  # 输出文件名  

# # 读取 Excel 文件  
# df = pd.read_excel(input_file)  

# # 要剔除的行索引（注意：pandas 的索引从 0 开始，而 Excel 行号从 1 开始）  
# rows_to_remove = [0, 2, 5, 8, 12]  # 对应 Excel 的 1, 3, 6, 9, 13 行  

# # 删除指定行  
# df_cleaned = df.drop(index=rows_to_remove)  

# # 将结果保存到新的 Excel 文件  
# df_cleaned.to_excel(output_file, index=False)  

# print(f"处理完成，剔除后的数据已保存到 {output_file}")
#######################################################################################################################################################
# # 加载完整模型  
# import torch  

# # 假设保存的文件名是 "model.pth"  
# model = torch.load("F_train_5paras4_30-[1112].pth")  

# # 查看模型的参数  
# for name, param in model.named_parameters():  
#     print(f"Parameter: {name}, Shape: {param.shape}")  
#     print(param.data)  # 打印具体的参数值
########################################################################################################################################################

# import torch  
# import torch.nn as nn  
# import torch.optim as optim  
# from torch.utils.data import DataLoader, Dataset  
# import numpy as np  

# # 模拟数据集  
# class RegressionDataset(Dataset):  
#     def __init__(self, num_samples=100, noise=0.1):  
#         self.x = torch.linspace(-3, 3, num_samples).unsqueeze(1)  
#         self.y = torch.sin(self.x) + noise * torch.randn(num_samples, 1)  
    
#     def __len__(self):  
#         return len(self.x)  
    
#     def __getitem__(self, idx):  
#         return self.x[idx], self.y[idx]  

# # 创建数据集  
# dataset = RegressionDataset()  
# train_loader = DataLoader(dataset, batch_size=16, shuffle=True)  
# test_loader = DataLoader(dataset, batch_size=16, shuffle=False)  

# # 定义贝叶斯神经网络（使用 Dropout）  
# class BayesianNN(nn.Module):  
#     def __init__(self, input_dim, hidden_dim, output_dim):  
#         super(BayesianNN, self).__init__()  
#         self.fc1 = nn.Linear(input_dim, hidden_dim)  
#         self.dropout = nn.Dropout(0.2)  # Dropout 模拟权重的不确定性  
#         self.fc2 = nn.Linear(hidden_dim, output_dim)  
    
#     def forward(self, x):  
#         x = torch.relu(self.fc1(x))  
#         x = self.dropout(x)  # 在训练和测试时都启用 Dropout  
#         x = self.fc2(x)  
#         return x  

# # 初始化模型  
# model = BayesianNN(input_dim=1, hidden_dim=50, output_dim=1)  
# criterion = nn.MSELoss()  
# optimizer = optim.Adam(model.parameters(), lr=0.01)  

# # 训练模型  
# def train_model(model, dataloader, criterion, optimizer, epochs=100):  
#     model.train()  
#     for epoch in range(epochs):  
#         for x, y in dataloader:  
#             optimizer.zero_grad()  
#             outputs = model(x)  
#             loss = criterion(outputs, y)  
#             loss.backward()  
#             optimizer.step()  
#         if (epoch + 1) % 10 == 0:  
#             print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")  

# train_model(model, train_loader, criterion, optimizer)

# # 测试模型并量化不确定性  
# def predict_with_uncertainty(f_model, x, n_iter=100):  
#     f_model.eval()  
#     preds = torch.zeros(n_iter, x.size(0))  
    
#     with torch.no_grad():  
#         for i in range(n_iter):  
#             preds[i] = f_model(x).squeeze()  # 多次前向传播  
    
#     # 计算均值和标准差  
#     pred_mean = preds.mean(dim=0)  
#     pred_std = preds.std(dim=0)  
#     return pred_mean, pred_std  

# # 测试数据  
# x_test = torch.linspace(-3, 3, 100).unsqueeze(1)  
# y_mean, y_std = predict_with_uncertainty(model, x_test)  

# # 可视化结果  
# import matplotlib.pyplot as plt  

# # 原始数据  
# plt.scatter(dataset.x.numpy(), dataset.y.numpy(), label="Data", color="blue", alpha=0.5)  

# # 预测均值  
# plt.plot(x_test.numpy(), y_mean.numpy(), label="Predictive Mean", color="red")  

# # 不确定性范围（均值 ± 2*标准差）  
# plt.fill_between(  
#     x_test.squeeze().numpy(),  
#     (y_mean - 2 * y_std).numpy(),  
#     (y_mean + 2 * y_std).numpy(),  
#     color="orange",  
#     alpha=0.3,  
#     label="Uncertainty (±2 std)"  
# )  

# plt.legend()  
# plt.show()
####################################################################################################################################################
# import torch  
# import torch.nn as nn  
# import torch.optim as optim  
# from torch.utils.data import DataLoader, Dataset  
# import pyro  
# import pyro.distributions as dist  
# from pyro.nn import PyroSample, PyroModule  
# from pyro.infer import SVI, Trace_ELBO  
# from pyro.optim import Adam  

# # 模拟数据集  
# class RegressionDataset(Dataset):  
#     def __init__(self, num_samples=1000):  
#         self.x = torch.rand(num_samples, 5, 1, 1)  # 输入形状为 (batch_size, 5, 1, 1)  
#         self.y = torch.rand(num_samples, 7)  # 输出形状为 (batch_size, 7)  
    
#     def __len__(self):  
#         return len(self.x)  
    
#     def __getitem__(self, idx):  
#         return self.x[idx], self.y[idx]  

# # 创建数据集  
# dataset = RegressionDataset()  
# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)  

# # 定义贝叶斯神经网络  
# class BayesianConvNet(PyroModule):  
#     def __init__(self):  
#         super(BayesianConvNet, self).__init__()  
        
#         # 卷积层  
#         self.conv1 = PyroModule[nn.Conv2d](in_channels=5,out_channels=16,kernel_size=3,padding=1)  
#         self.conv2 = PyroModule[nn.Conv2d](in_channels=16,out_channels=32,kernel_size=3,padding=1)  
        
#         # 全连接层  
#         self.fc1 = PyroModule[nn.Linear](32,64)  
#         self.fc2 = PyroModule[nn.Linear](64,7)  
        
#         # 激活函数  
#         self.relu = nn.ReLU()  
#         self.flatten = nn.Flatten()  

#         # 定义权重和偏置的先验分布  
#         self.conv1.weight = PyroSample(dist.Normal(0., 1.).expand([16, 5, 3, 3]).to_event(4))  
#         self.conv1.bias = PyroSample(dist.Normal(0., 1.).expand([16]).to_event(1))  
#         self.conv2.weight = PyroSample(dist.Normal(0., 1.).expand([32, 16, 3, 3]).to_event(4))  
#         self.conv2.bias = PyroSample(dist.Normal(0., 1.).expand([32]).to_event(1))  
#         self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([64, 32]).to_event(2))  
#         self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([64]).to_event(1))  
#         self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([7, 64]).to_event(2))  
#         self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([7]).to_event(1))  

#     def forward(self, x, y=None):  
#         # 卷积层 + 激活函数  
#         x = self.relu(self.conv1(x))  
#         x = self.relu(self.conv2(x))  
        
#         # 展平  
#         x = self.flatten(x)  
        
#         # 全连接层 + 激活函数  
#         x = self.relu(self.fc1(x))  
#         x = self.fc2(x)  # 输出层  
        
#         # 定义观测分布  
#         with pyro.plate("data", x.size(0)):  
#             obs = pyro.sample("obs", dist.Normal(x, 0.1).to_event(1), obs=y)  
#         return x  

# # 初始化模型  
# model = BayesianConvNet()  
# for name, param in model.named_parameters():  
#     print(f"Parameter name: {name}, Shape: {param.shape}")
# from pyro.nn import PyroParam  
# # 定义导向函数（Guide）  
# class Guide(PyroModule):  
#     def __init__(self):  
#         super(Guide, self).__init__()  
#         self.conv1_weight_loc = PyroParam(torch.randn(16, 5, 3, 3))  
#         self.conv1_weight_scale = PyroParam(torch.ones(16, 5, 3, 3), constraint=dist.constraints.positive)  
#         self.conv1_bias_loc = PyroParam(torch.randn(16))  
#         self.conv1_bias_scale = PyroParam(torch.ones(16), constraint=dist.constraints.positive)  
        
#         self.conv2_weight_loc = PyroParam(torch.randn(32, 16, 3, 3))  
#         self.conv2_weight_scale = PyroParam(torch.ones(32, 16, 3, 3), constraint=dist.constraints.positive)  
#         self.conv2_bias_loc = PyroParam(torch.randn(32))  
#         self.conv2_bias_scale = PyroParam(torch.ones(32), constraint=dist.constraints.positive)  
        
#         self.fc1_weight_loc = PyroParam(torch.randn(64, 32))  
#         self.fc1_weight_scale = PyroParam(torch.ones(64, 32), constraint=dist.constraints.positive)  
#         self.fc1_bias_loc = PyroParam(torch.randn(64))  
#         self.fc1_bias_scale = PyroParam(torch.ones(64), constraint=dist.constraints.positive)  
        
#         self.fc2_weight_loc = PyroParam(torch.randn(7, 64))  
#         self.fc2_weight_scale = PyroParam(torch.ones(7, 64), constraint=dist.constraints.positive)  
#         self.fc2_bias_loc = PyroParam(torch.randn(7))  
#         self.fc2_bias_scale = PyroParam(torch.ones(7), constraint=dist.constraints.positive)  

#     def forward(self, x, y=None):  
#         pyro.sample("conv1.weight", dist.Normal(self.conv1_weight_loc, self.conv1_weight_scale).to_event(4))  
#         pyro.sample("conv1.bias", dist.Normal(self.conv1_bias_loc, self.conv1_bias_scale).to_event(1))  
#         pyro.sample("conv2.weight", dist.Normal(self.conv2_weight_loc, self.conv2_weight_scale).to_event(4))  
#         pyro.sample("conv2.bias", dist.Normal(self.conv2_bias_loc, self.conv2_bias_scale).to_event(1))  
#         pyro.sample("fc1.weight", dist.Normal(self.fc1_weight_loc, self.fc1_weight_scale).to_event(2))  
#         pyro.sample("fc1.bias", dist.Normal(self.fc1_bias_loc, self.fc1_bias_scale).to_event(1))  
#         pyro.sample("fc2.weight", dist.Normal(self.fc2_weight_loc, self.fc2_weight_scale).to_event(2))  
#         pyro.sample("fc2.bias", dist.Normal(self.fc2_bias_loc, self.fc2_bias_scale).to_event(1))  

# guide = Guide()  

# # 定义优化器和推断算法  
# optimizer = Adam({"lr": 0.01})  
# svi = SVI(model, guide, optimizer, loss=Trace_ELBO())  

# # 训练模型  
# def train_model(model, guide, dataloader, svi, epochs=10):  
#     for epoch in range(epochs):  
#         total_loss = 0  
#         for x, y in dataloader:  
#             loss = svi.step(x, y)  
#             total_loss += loss  
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")  

# # 开始训练  
# train_model(model, guide, train_loader, svi)
###############################################################################################################################################
# import pyro  
# import pyro.distributions as dist  
# import pyro.poutine as poutine  

# # 定义一个简单模型  
# def model():  
#     x = pyro.sample("c", dist.Normal(0, 1))  # 定义随机变量 x  
#     y = pyro.sample("d", dist.Normal(x, 1))  # 定义随机变量 y  
#     return y  

# # 使用 poutine.trace 捕获模型中的随机变量  
# trace = poutine.trace(model).get_trace()  

# # 访问随机变量的值  
# x_value = trace.nodes["x"]["value"]  # 获取随机变量 x 的值  
# y_value = trace.nodes["y"]["value"]  # 获取随机变量 y 的值  

# print(f"x: {x_value}, y: {y_value}")
#####################################################################################################################################################
# import torch  
# import torch.nn as nn  
# from F_model_8paras import F_Net_2D
  

# # 初始化网络  
# # net = F_Net_2D()  
# Project_folder = "D:\\keti\\zjh\\wl\\" 
# model_pth = "F_train_5paras3_2000-[1112].pth"
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
# net = torch.load(Project_folder  + model_pth, map_location=DEVICE)
# # 查看网络中的参数  
# for name, param in net.named_parameters():  
#     print(f"Parameter name: {name}, Shape: {param}")

######################################################################################################################################################
# import torch  
# import torch.nn as nn  

# # 定义一个简单的神经网络  
# class SimpleNet(nn.Module):  
#     def __init__(self):  
#         super(SimpleNet, self).__init__()  
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)  
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)  
#         self.fc1 = nn.Linear(32 * 6 * 6, 128)  # 假设输入图像大小为 8x8  
#         self.fc2 = nn.Linear(128, 10)  

#     def forward(self, x):  
#         x = torch.relu(self.conv1(x))  
#         x = torch.relu(self.conv2(x))  
#         x = torch.flatten(x, 1)  
#         x = torch.relu(self.fc1(x))  
#         x = self.fc2(x)  
#         return x  

# # 初始化网络  
# model = SimpleNet()  

# # 打印网络中所有权重的名称和形状  
# print("Model Parameters:")  
# for name, param in model.named_parameters():  
#     print(f"Name: {name}, Shape: {param.shape}")
##############################################################################################################################################
# import torch  
# import torch.nn as nn  
# import pyro  
# import pyro.distributions as dist  
# from pyro.nn import PyroSample  
# from pyro.infer import SVI, Trace_ELBO  
# from pyro.optim import Adam  

# # 定义一个两层卷积网络  
# class ConvNet(nn.Module):  
#     def __init__(self):  
#         super(ConvNet, self).__init__()  
#         # 第一层卷积：输入通道 3，输出通道 16，卷积核大小 3x3  
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)  
#         # 第二层卷积：输入通道 16，输出通道 32，卷积核大小 3x3  
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)  
#         # 全连接层：输入 32*6*6，输出 90  
#         self.fc = nn.Linear(32 * 6 * 6, 90)  

#     def forward(self, x):  
#         x = torch.relu(self.conv1(x))  
#         x = torch.relu(self.conv2(x))  
#         x = torch.flatten(x, 1)  # 展平  
#         x = self.fc(x)  
#         return x  

# # 初始化网络  
# base_model = ConvNet()  

# # 定义先验分布  
# def model_prior():  
#     priors = {}  
#     for name, param in base_model.named_parameters():  
#         priors[name] = dist.Normal(0., 1.).expand(param.shape).to_event(param.dim())  
#     return priors  

# # 使用 pyro.random_module 将网络参数替换为随机变量  
# bayesian_model = pyro.random_module("bayesian_convnet", base_model, model_prior())  

# # 定义模型  
# def model(x, y=None):  
#     sampled_model = bayesian_model()  # 从随机模块中采样一个模型  
#     mean = sampled_model(x)  # 前向传播  
#     # 定义观测分布  
#     with pyro.plate("data", x.size(0)):  
#         pyro.sample("obs", dist.Normal(mean, 1.0).to_event(1), obs=y)  

# # 定义导向模型（guide）  
# def guide(x, y=None):  
#     # 定义每个参数的变分分布（均值和标准差）  
#     variational_params = {}  
#     for name, param in base_model.named_parameters():  
#         loc = pyro.param(f"{name}_loc", torch.randn_like(param))  # 均值  
#         scale = pyro.param(f"{name}_scale", torch.ones_like(param), constraint=dist.constraints.positive)  # 标准差  
#         variational_params[name] = dist.Normal(loc, scale).to_event(param.dim())  
    
#     # 使用 pyro.random_module 将变分分布应用到模型  
#     bayesian_guide = pyro.random_module("bayesian_convnet", base_model, variational_params)  
#     return bayesian_guide()  

# # 检查是否有 GPU 并将模型移动到 GPU  
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
# base_model.to(device)  

# # 生成一些随机数据  
# x_data = torch.randn(100, 3, 32, 32).to(device)  # 100 个样本，每个样本是 32x32 的 RGB 图像  
# y_data = torch.randn(100, 90).to(device)  # 对应的输出是 90 个参数  

# # 定义优化器和推断方法  
# optimizer = Adam({"lr": 0.01})  
# svi = SVI(model, guide, optimizer, loss=Trace_ELBO())  

# # 训练  
# num_steps = 1000  
# for step in range(num_steps):  
#     loss = svi.step(x_data, y_data)  
#     if step % 100 == 0:  
#         print(f"Step {step} - Loss: {loss:.4f}")  

# # 测试网络  
# x_test = torch.randn(5, 3, 32, 32).to(device)  # 测试数据  
# sampled_model = bayesian_model()  # 从随机模块中采样一个模型  
# sampled_model.to(device)  
# output = sampled_model(x_test)  
# print("Output shape:", output.shape)
# import torch  
# import torch.nn as nn  
# import pyro  
# import pyro.distributions as dist  
# from pyro.infer import SVI, Trace_ELBO  
# from pyro.optim import Adam  

# # 定义神经网络模型  
# class BNN(nn.Module):  
#     def __init__(self):  
#         super(BNN, self).__init__()  
#         # 两层卷积层  
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # 输入通道 1，输出通道 16  
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 输入通道 16，输出通道 32  
        
#         # 四层全连接层  
#         self.fc1 = nn.Linear(32 * 5 * 5, 128)  # 假设输入经过卷积后展平为 32*5*5  
#         self.fc2 = nn.Linear(128, 64)  
#         self.fc3 = nn.Linear(64, 32)  
#         self.fc4 = nn.Linear(32, 7)  # 输出 7 个参数  

#     def forward(self, x):  
#         # 卷积层 + 激活函数 + 池化  
#         x = torch.relu(self.conv1(x))  
#         x = torch.relu(self.conv2(x))  
#         x = torch.flatten(x, start_dim=1)  # 展平  
        
#         # 全连接层 + 激活函数  
#         x = torch.relu(self.fc1(x))  
#         x = torch.relu(self.fc2(x))  
#         x = torch.relu(self.fc3(x))  
#         x = self.fc4(x)  # 输出层  
#         return x  

# # 定义先验分布  
# def get_priors(model):  
#     priors = {}  
#     for name, param in model.named_parameters():  
#         # 为每个参数定义正态分布先验  
#         priors[name] = dist.Normal(torch.zeros_like(param), torch.ones_like(param)).to_event(param.dim())  
#     return priors  

# # 定义模型函数  
# def model(x, y=None):  
#     # 初始化神经网络  
#     bnn = BNN()  
#     # 定义先验分布  
#     priors = get_priors(bnn)  
#     # 使用 pyro.random_module 将模型参数与先验分布关联  
#     lifted_bnn = pyro.random_module("bnn", bnn, priors)  
#     sampled_model = lifted_bnn()  # 从先验中采样参数  
#     # 前向传播  
#     output = sampled_model(x)  
#     # 定义观测分布  
#     with pyro.plate("data", x.size(0)):  
#         pyro.sample("obs", dist.Normal(output, 1.0).to_event(1), obs=y)  

# # 定义导向模型（guide）  
# def guide(x, y=None):  
#     # 初始化神经网络  
#     bnn = BNN()  
#     # 定义变分分布（与先验分布相同的结构，但参数可学习）  
#     priors = get_priors(bnn)  
#     lifted_bnn = pyro.random_module("bnn", bnn, priors)  
#     return lifted_bnn()  

# # 数据生成  
# def generate_data(num_samples=100):  
#     x = torch.randn(num_samples, 1, 5, 5)  # 输入为 5x5 的单通道图像  
#     y = torch.randn(num_samples, 7)  # 输出为 7 个参数  
#     return x, y  

# # 训练  
# def train():  
#     # 生成数据  
#     x_data, y_data = generate_data()  
    
#     # 定义优化器和推断方法  
#     optimizer = Adam({"lr": 0.01})  
#     svi = SVI(model, guide, optimizer, loss=Trace_ELBO())  
    
#     # 训练循环  
#     num_steps = 1000  
#     for step in range(num_steps):  
#         loss = svi.step(x_data, y_data)  
#         if step % 100 == 0:  
#             print(f"Step {step} - Loss: {loss:.4f}")  

# # 测试  
# def test():  
#     # 生成测试数据  
#     x_test, _ = generate_data(num_samples=10)  
#     # 初始化神经网络  
#     bnn = BNN()  
#     # 定义先验分布  
#     priors = get_priors(bnn)  
#     # 使用 pyro.random_module  
#     lifted_bnn = pyro.random_module("bnn", bnn, priors)  
#     sampled_model = lifted_bnn()  # 从先验中采样参数  
#     # 前向传播  
#     output = sampled_model(x_test)  
#     print("Test Output:", output)  

# # 主函数  
# if __name__ == "__main__":  
#     pyro.clear_param_store()  # 清除 Pyro 参数存储器  
#     train()  # 训练模型  
#     test()   # 测试模型


# import torch  
# import torch.nn as nn  
# import pyro  
# import pyro.distributions as dist  
# from pyro.infer import SVI, Trace_ELBO  
# from pyro.optim import Adam  


# from test_model import BayesianNN


# # 定义模型函数  
# def model(x, y=None):  
#     # 定义先验分布  
#     bnn = BayesianNN()  
#     priors = {  
#         "conv1.weight": dist.Normal(torch.zeros_like(bnn.conv1.weight), torch.ones_like(bnn.conv1.weight)).to_event(4),  
#         "conv1.bias": dist.Normal(torch.zeros_like(bnn.conv1.bias), torch.ones_like(bnn.conv1.bias)).to_event(1),  
#         "fc1.weight": dist.Normal(torch.zeros_like(bnn.fc1.weight), torch.ones_like(bnn.fc1.weight)).to_event(2),  
#         "fc1.bias": dist.Normal(torch.zeros_like(bnn.fc1.bias), torch.ones_like(bnn.fc1.bias)).to_event(1),  
#     }  
    
#     # 使用 pyro.random_module 加载模型  
#     lifted_bnn = pyro.random_module("bnn", bnn, priors)  
#     sampled_model = lifted_bnn()  
    
#     # 模型输出  
#     output = sampled_model(x)  
    
#     # 定义观测分布  
#     with pyro.plate("data", x.size(0)):  
#         pyro.sample("obs", dist.Normal(output, 1.0).to_event(1), obs=y)  


# # 定义导向模型（guide）  
# def guide(x, y=None):  
#     # 定义变分分布的参数  
#     bnn = BayesianNN()  
#     priors = {  
#         "conv1.weight": dist.Normal(  
#             pyro.param("conv1_weight_loc", torch.randn_like(bnn.conv1.weight)),  
#             pyro.param("conv1_weight_scale", torch.ones_like(bnn.conv1.weight), constraint=dist.constraints.positive)  
#         ).to_event(4),  
#         "conv1.bias": dist.Normal(  
#             pyro.param("conv1_bias_loc", torch.randn_like(bnn.conv1.bias)),  
#             pyro.param("conv1_bias_scale", torch.ones_like(bnn.conv1.bias), constraint=dist.constraints.positive)  
#         ).to_event(1),  
#         "fc1.weight": dist.Normal(  
#             pyro.param("fc1_weight_loc", torch.randn_like(bnn.fc1.weight)),  
#             pyro.param("fc1_weight_scale", torch.ones_like(bnn.fc1.weight), constraint=dist.constraints.positive)  
#         ).to_event(2),  
#         "fc1.bias": dist.Normal(  
#             pyro.param("fc1_bias_loc", torch.randn_like(bnn.fc1.bias)),  
#             pyro.param("fc1_bias_scale", torch.ones_like(bnn.fc1.bias), constraint=dist.constraints.positive)  
#         ).to_event(1),  
#     }  
    
#     # 使用 pyro.random_module 加载模型  
#     lifted_bnn = pyro.random_module("bnn", bnn, priors)  
#     return lifted_bnn()  


# # 数据生成函数  
# def generate_data(num_samples=200):  
#     x = torch.randn(num_samples, 1, 5, 5)  # 输入为 [batch_size, channels, height, width]  
#     y = torch.randn(num_samples, 7)  # 输出为 [batch_size, output_dim]  
#     return x, y  


# # 训练函数  
# def train():  
#     # 生成数据  
#     x_data, y_data = generate_data()  
    
#     # 定义优化器和推断方法  
#     optimizer = Adam({"lr": 0.01})  
#     svi = SVI(model, guide, optimizer, loss=Trace_ELBO())  
    
#     # 训练循环  
#     num_steps = 1000  
#     for step in range(num_steps):  
#         loss = svi.step(x_data, y_data)  
#         if step % 100 == 0:  
#             print(f"Step {step} - Loss: {loss:.4f}")  


# # 测试函数  
# def test():  
#     x_test, _ = generate_data(num_samples=10)  
#     bnn = BayesianNN()  
#     priors = {  
#         "conv1.weight": dist.Normal(  
#             pyro.param("conv1_weight_loc"),  
#             pyro.param("conv1_weight_scale")  
#         ).to_event(4),  
#         "conv1.bias": dist.Normal(  
#             pyro.param("conv1_bias_loc"),  
#             pyro.param("conv1_bias_scale")  
#         ).to_event(1),  
#         "fc1.weight": dist.Normal(  
#             pyro.param("fc1_weight_loc"),  
#             pyro.param("fc1_weight_scale")  
#         ).to_event(2),  
#         "fc1.bias": dist.Normal(  
#             pyro.param("fc1_bias_loc"),  
#             pyro.param("fc1_bias_scale")  
#         ).to_event(1),  
#     }  
    
#     lifted_bnn = pyro.random_module("bnn", bnn, priors)  
#     sampled_model = lifted_bnn()  
#     output = sampled_model(x_test)  
#     print("Test Output:", output)  


# # 主函数  
# if __name__ == "__main__":  
#     pyro.clear_param_store()  
#     train()  
#     test()

#######################################################################################################################
# import pyro  
# import torch  
# from F_model_8paras import F_Net_1D
# from torch.utils.data import DataLoader, Dataset  
# from F_Dataset_8paras import MH_Data
# # 假设 `guide` 是你的导向模型  
# # 提取 guide 中的参数  
# Project_folder = "D:\\keti\\zjh\\wl\\" 

# mh_val = MH_Data(Project_folder + 'test.xlsx')

# val_loader = DataLoader(batch_size=10, dataset=mh_val, shuffle=True,
#                             num_workers=1, pin_memory=True, persistent_workers=True)
# net =F_Net_1D()
# pyro.get_param_store().load("guide_params.pt")  

# # 打印所有参数  
# param_store = pyro.get_param_store()  
# # for name, value in param_store.items():  
# #     print(f"Parameter name: {name}")  
# priors = {}  

# # 遍历参数存储器中的所有参数  
# for name, value in param_store.items():  
#     # 将参数名称作为字典的键，参数值作为字典的值  
#     priors[name] = value 
#     # print(f"Parameter value: {value}") 

# # 打印结果  
# # for key, val in priors.items():  
#     # print(f"'{key}': {key},")  

# lifted_bnn = pyro.random_module("bnn", net, priors)() 
# if __name__ == '__main__':
#     for x, y in val_loader:  
#         print(x.shape)
#         output = lifted_bnn(x)  
#         print("Model output:", output)  
#         print("real output:", y)
######################################################################################################################
# import torch  
# import torch.nn as nn  
# import pyro  
# import pyro.distributions as dist  
# from pyro.infer import SVI, Trace_ELBO  
# from pyro.optim import Adam  

# # 检查是否有可用的 GPU  
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
# print(f"Using device: {device}")  

# # 定义神经网络  
# class BayesianNN(nn.Module):  
#     def __init__(self):  
#         super(BayesianNN, self).__init__()  
#         self.fc1 = nn.Linear(10, 20)  # 输入大小为 10，隐藏层大小为 20  
#         self.fc2 = nn.Linear(20, 1)  # 隐藏层大小为 20，输出大小为 1  

#     def forward(self, x):  
#         x = torch.relu(self.fc1(x))  
#         x = self.fc2(x)  
#         return x  

# # 定义 Pyro 模型  
# def model(x_data, y_data):  
#     # 定义先验分布  
#     priors = {  
#         "fc1.weight": dist.Normal(torch.zeros_like(bnn.fc1.weight), torch.ones_like(bnn.fc1.weight)).to_event(2),  
#         "fc1.bias": dist.Normal(torch.zeros_like(bnn.fc1.bias), torch.ones_like(bnn.fc1.bias)).to_event(1),  
#         "fc2.weight": dist.Normal(torch.zeros_like(bnn.fc2.weight), torch.ones_like(bnn.fc2.weight)).to_event(2),  
#         "fc2.bias": dist.Normal(torch.zeros_like(bnn.fc2.bias), torch.ones_like(bnn.fc2.bias)).to_event(1),  
#     }  

#     # 使用 pyro.random_module 将先验分布与模型参数绑定  
#     lifted_module = pyro.random_module("bnn", bnn, priors)  
#     lifted_bnn = lifted_module()  

#     # 前向传播  
#     with pyro.plate("data", x_data.size(0)):  
#         prediction_mean = lifted_bnn(x_data)  
#         pyro.sample("obs", dist.Normal(prediction_mean, 0.1).to_event(1), obs=y_data)  

# # 定义 Pyro 的 guide（变分分布）  
# def guide(x_data, y_data):  
#     # 定义变分分布的参数  
#     fc1_weight_loc = pyro.param("fc1_weight_loc", torch.randn_like(bnn.fc1.weight).to(device))  
#     fc1_weight_scale = pyro.param("fc1_weight_scale", torch.ones_like(bnn.fc1.weight).to(device), constraint=dist.constraints.positive)  
#     fc1_bias_loc = pyro.param("fc1_bias_loc", torch.randn_like(bnn.fc1.bias).to(device))  
#     fc1_bias_scale = pyro.param("fc1_bias_scale", torch.ones_like(bnn.fc1.bias).to(device), constraint=dist.constraints.positive)  

#     fc2_weight_loc = pyro.param("fc2_weight_loc", torch.randn_like(bnn.fc2.weight).to(device))  
#     fc2_weight_scale = pyro.param("fc2_weight_scale", torch.ones_like(bnn.fc2.weight).to(device), constraint=dist.constraints.positive)  
#     fc2_bias_loc = pyro.param("fc2_bias_loc", torch.randn_like(bnn.fc2.bias).to(device))  
#     fc2_bias_scale = pyro.param("fc2_bias_scale", torch.ones_like(bnn.fc2.bias).to(device), constraint=dist.constraints.positive)  

#     # 定义变分分布  
#     priors = {  
#         "fc1.weight": dist.Normal(fc1_weight_loc, fc1_weight_scale).to_event(2),  
#         "fc1.bias": dist.Normal(fc1_bias_loc, fc1_bias_scale).to_event(1),  
#         "fc2.weight": dist.Normal(fc2_weight_loc, fc2_weight_scale).to_event(2),  
#         "fc2.bias": dist.Normal(fc2_bias_loc, fc2_bias_scale).to_event(1),  
#     }  

#     # 使用 pyro.random_module 将变分分布与模型参数绑定  
#     lifted_module = pyro.random_module("bnn", bnn, priors)  
#     return lifted_module()  

# # 创建数据集  
# def generate_data(n_samples=100):  
#     x = torch.randn(n_samples, 10).to(device)  # 输入大小为 10  
#     y = torch.sum(x, dim=1, keepdim=True) + 0.1 * torch.randn(n_samples, 1).to(device)  # 简单的线性关系  
#     return x, y  

# # 生成训练数据  
# x_train, y_train = generate_data(1000)  

# # 创建 BNN 实例并移动到 GPU  
# bnn = BayesianNN().to(device)  

# # 定义优化器和损失函数  
# optimizer = Adam({"lr": 0.01})  
# svi = SVI(model, guide, optimizer, loss=Trace_ELBO())  

# # 训练模型  
# num_iterations = 5000  
# for step in range(num_iterations):  
#     loss = svi.step(x_train, y_train)  
#     if step % 500 == 0:  
#         print(f"Step {step} - Loss: {loss}")  

# # 打印训练后的参数  
# for name, value in pyro.get_param_store().items():  
#     print(f"{name}: {value}")  

# # 测试模型  
# x_test, y_test = generate_data(100)  
# bnn.eval()  
# with torch.no_grad():  
#     lifted_bnn = guide(x_test, y_test)  # 从 guide 中采样模型  
#     predictions = lifted_bnn(x_test)  
#     print("Predictions:", predictions[:5])  
#     print("Ground Truth:", y_test[:5])

# import torch  
# import torch.nn as nn  
# import pyro  
# import pyro.distributions as dist  

# # 设置随机种子  
# pyro.set_rng_seed(42)  

# # 定义一个简单的神经网络  
# class SimpleNet(nn.Module):  
#     def __init__(self):  
#         super(SimpleNet, self).__init__()  
#         self.fc1 = nn.Linear(2, 1)  # 输入维度为 2，输出维度为 1  

#     def forward(self, x):  
#         return self.fc1(x)  

# # 创建网络实例  
# net = SimpleNet()  

# # 定义先验分布  
# priors = {  
#     "fc1.weight": dist.Normal(torch.zeros_like(net.fc1.weight), torch.ones_like(net.fc1.weight)).to_event(1),  
#     "fc1.bias": dist.Normal(torch.zeros_like(net.fc1.bias), torch.ones_like(net.fc1.bias)).to_event(1),  
# }  

# # 使用 pyro.random_module 构建随机网络  
# bnn = pyro.random_module("bnn", net, priors)  

# print(priors)
# # 随机化网络  
# randomized_net = bnn()  

# # 打印随机化网络的参数值  
# print("Randomized Network Parameters:")  
# for name, param in randomized_net.named_parameters():  
#     print(f"Parameter name: {name}")  
#     print(f"Randomized value: {param.data}")  

#     # 从 priors 中采样值  
#     if name in priors:  
#         sampled_value = priors[name].sample()  
#         print(f"Sampled from prior: {sampled_value}")

import numpy as np  
from scipy.spatial.distance import euclidean  
from scipy.stats import pearsonr  
from scipy.spatial.distance import cosine  
from fastdtw import fastdtw  

# 生成示例数据：四条曲线  
x = np.linspace(0, 10, 100)  
curve1 = np.sin(x)  
curve2 = np.sin(x + 0.1)  # 与 curve1 相似  
curve3 = np.cos(x)        # 与 curve1 不太相似  
curve4 = np.sin(x + 1)    # 与 curve1 不太相似  

curves = [curve1, curve2, curve3, curve4]  

# 计算相似度  
def calculate_similarity(curves):  
    num_curves = len(curves)  
    similarity_matrix = np.zeros((num_curves, num_curves))  

    for i in range(num_curves):  
        for j in range(num_curves):  
            if i == j:  
                similarity_matrix[i][j] = 1  # 自身相似度为 1  
            else:  
                # 1. 计算欧几里得距离  
                euclidean_distance = euclidean(curves[i], curves[j])  
                similarity_matrix[i][j] = 1 / (1 + euclidean_distance)  # 距离越小，相似度越高  

                # 2. 计算余弦相似度  
                cosine_similarity = 1 - cosine(curves[i], curves[j])  
                similarity_matrix[i][j] = max(similarity_matrix[i][j], cosine_similarity)  

                # 3. 计算皮尔逊相关系数  
                pearson_corr, _ = pearsonr(curves[i], curves[j])  
                similarity_matrix[i][j] = max(similarity_matrix[i][j], pearson_corr)  

                # 4. 动态时间规整（DTW）  
                dtw_distance, _ = fastdtw(curves[i], curves[j])  
                dtw_similarity = 1 / (1 + dtw_distance)  
                similarity_matrix[i][j] = max(similarity_matrix[i][j], dtw_similarity) 
                upper_triangle_indices = np.triu_indices(len(curves), k=1)  
                average_similarity = np.mean(similarity_matrix[upper_triangle_indices])   

    return average_similarity  

# 计算相似度矩阵  
average_similarity = calculate_similarity(curves)  


# 打印整体相似度  
print(f'Overall Similarity: {average_similarity:.4f}')