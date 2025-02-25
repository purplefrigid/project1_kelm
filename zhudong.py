# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from torch.utils.data import Dataset


# # 数据预处理
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# # 加载Mnist数据集
# trainset = torchvision.datasets.MNIST('data/', download=True, train=True, transform=transform)
# testset = torchvision.datasets.MNIST('data/', download=True, train=False, transform=transform)

# # 划分初始训练集和主动学习集
# initial_train_size = int(0.2 * len(trainset))
# initial_trainset, active_learning_set = torch.utils.data.random_split(trainset, [initial_train_size,
#                                                                                  len(trainset) - initial_train_size])

# # 创建数据加载器
# initial_trainloader = DataLoader(initial_trainset, batch_size=64, shuffle=True)
# active_learning_loader = DataLoader(active_learning_set, batch_size=64, shuffle=False)
# testloader = DataLoader(testset, batch_size=64, shuffle=False)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from torch.utils.data import Dataset

# def train_model(model, criterion, optimizer, dataloader, num_epochs):
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for i, data in enumerate(dataloader, 0):
#             inputs, labels = data
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}')


# def model_test(testloader, net):
#     correct = 0  
#     total = 0  
#     with torch.no_grad():  
#         for data in testloader:
#             images, labels = data  
#             outputs = net(images) 
#             _, predicted = torch.max(outputs.data, 1)  
#             total += labels.size(0) 
#             correct += (predicted == labels).sum().item()  
#     print(f'模型在测试集上的准确率: {100 * correct / total} %')  

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout2d(0.25)
#         self.dropout2 = nn.Dropout2d(0.5)
#         self.fc1 = nn.Linear(9216, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = nn.functional.relu(x)
#         x = self.conv2(x)
#         x = nn.functional.relu(x)
#         x = nn.functional.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = nn.functional.relu(x)
#         x = self.dropout2(x)
#         output = self.fc2(x)
#         return output

# # 实例化模型和优化器
# net = Net()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.001)

# # 调用train_model函数进行初始训练
# num_epochs = 2
# train_model(net, criterion, optimizer, initial_trainloader, num_epochs)

# # 调用model_test函数进行模型测试
# model_test(testloader, net)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from torch.utils.data import Dataset

# # 主动学习
# num_active_learning_rounds = 5 
# for round in range(num_active_learning_rounds):  
#     # 选择不确定性最高的样本
#     uncertainties = get_uncertainty(net, active_learning_loader)
#     indices_to_label = torch.argsort(torch.tensor(uncertainties), descending=True)[:1000]  # 假设每次选择1000个样本
#     labeled_dataset = SubsetDataset(active_learning_set, indices_to_label)

#     # 更新训练集
#     updated_trainset = torch.utils.data.ConcatDataset([initial_trainset, labeled_dataset])
#     trainloader = DataLoader(updated_trainset, batch_size=64, shuffle=True)

#     # 重新训练模型
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for i, data in enumerate(trainloader, 0):
#             inputs, labels = data
#             optimizer.zero_grad()
#             outputs = net(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         print(f'Active Learning Round {round + 1}, Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')
######################################################################################################################################################################
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import WhiteKernel, RBF
# from modAL.models import ActiveLearner
# # 生成回归数据集
# X = np.random.choice(np.linspace(0, 20, 10000), size=200, replace=False).reshape(-1, 1)
# y = np.sin(X) + np.random.normal(scale=0.3, size=X.shape)

# with plt.style.context('seaborn-v0_8'):
#     plt.figure(figsize=(10, 5))
#     plt.scatter(X, y, c='k', s=20)
#     plt.title('sin(x) + noise')
#     plt.show()

# n_initial = 5  # 最开始随机选取5个标注好的数据集进行训练
# initial_idx = np.random.choice(range(len(X)), size=n_initial, replace=False)
# X_training, y_training = X[initial_idx], y[initial_idx]

# def GP_regression_std(regressor, X):
#     _, std = regressor.predict(X, return_std=True)  # 不确定度度量
#     query_idx = np.argmax(std)  # 样本的选取
#     return query_idx, X[query_idx]

# kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))

# regressor = ActiveLearner(
#     estimator=GaussianProcessRegressor(kernel=kernel),
#     query_strategy=GP_regression_std,
#     X_training=X_training.reshape(-1, 1), y_training=y_training.reshape(-1, 1)
# )
# X_grid = np.linspace(0, 20, 1000)
# y_pred, y_std = regressor.predict(X_grid.reshape(-1, 1), return_std=True)
# y_pred, y_std = y_pred.ravel(), y_std.ravel()
# with plt.style.context('seaborn-v0_8'):
#     plt.figure(figsize=(10, 5))
#     plt.plot(X_grid, y_pred)
#     plt.fill_between(X_grid, y_pred - y_std, y_pred + y_std, alpha=0.2)
#     plt.scatter(X, y, c='k', s=20)
#     plt.title('Initial prediction')
#     plt.show()

# n_queries = 10
# for idx in range(n_queries):
#     query_idx, query_instance = regressor.query(X)
#     regressor.teach(X[query_idx].reshape(1, -1), y[query_idx].reshape(1, -1))

# y_pred_final, y_std_final = regressor.predict(X_grid.reshape(-1, 1), return_std=True)
# y_pred_final, y_std_final = y_pred_final.ravel(), y_std_final.ravel()

# with plt.style.context('seaborn-v0_8'):
#     plt.figure(figsize=(10, 8))
#     plt.plot(X_grid, y_pred_final)
#     plt.fill_between(X_grid, y_pred_final - y_std_final, y_pred_final + y_std_final, alpha=0.2)
#     plt.scatter(X, y, c='k', s=20)
#     plt.title('Prediction after active learning')
#     plt.show()

#########################################################################################################################################################################
# from copy import deepcopy

# import matplotlib.pyplot as plt
# import numpy as np
# from modAL.models import ActiveLearner, Committee
# from sklearn.datasets import load_iris
# from sklearn.decomposition import PCA
# from sklearn.ensemble import RandomForestClassifier

# # loading the iris dataset
# iris = load_iris()

# # visualizing the classes
# with plt.style.context('seaborn-v0_8'):
#     plt.figure(figsize=(7, 7))
#     pca = PCA(n_components=2).fit_transform(iris['data'])
#     plt.scatter(x=pca[:, 0], y=pca[:, 1], c=iris['target'], cmap='viridis', s=50)
#     plt.title('The iris dataset')
#     plt.show()

# X_pool = deepcopy(iris['data'])
# y_pool = deepcopy(iris['target'])

# n_members = 2
# learner_list = list()

# for member_idx in range(n_members):
#     # initial training data
#     n_initial = 5
#     train_idx = np.random.choice(range(X_pool.shape[0]), size=n_initial, replace=False)
#     X_train = X_pool[train_idx]
#     y_train = y_pool[train_idx]

#     # creating a reduced copy of the data with the known instances removed
#     X_pool = np.delete(X_pool, train_idx, axis=0)
#     y_pool = np.delete(y_pool, train_idx)

#     # initializing learner
#     learner = ActiveLearner(
#         estimator=RandomForestClassifier(),
#         X_training=X_train, y_training=y_train
#     )
#     learner_list.append(learner)


# committee = Committee(learner_list=learner_list)

# # visualizing the initial predictions
# with plt.style.context('seaborn-v0_8'):
#     plt.figure(figsize=(7, 7))
#     prediction = committee.predict(iris['data'])
#     plt.scatter(x=pca[:, 0], y=pca[:, 1], c=prediction, cmap='viridis', s=50)
#     plt.title('Committee initial predictions, accuracy = %1.3f' % committee.score(iris['data'], iris['target']))
#     plt.show()

# # query by committee
#     n_queries = 10
#     for idx in range(n_queries):
#         query_idx, query_instance = committee.query(X_pool)
#         committee.teach(
#             X=X_pool[query_idx].reshape(1, -1),
#             y=y_pool[query_idx].reshape(1, )
#         )
#         # remove queried instance from pool
#         X_pool = np.delete(X_pool, query_idx, axis=0)
#         y_pool = np.delete(y_pool, query_idx)

# with plt.style.context('seaborn-v0_8'):
#     plt.figure(figsize=(7, 7))
#     prediction = committee.predict(iris['data'])
#     plt.scatter(x=pca[:, 0], y=pca[:, 1], c=prediction, cmap='viridis', s=50)
#     plt.title('Committee predictions after %d queries, accuracy = %1.3f'
#               % (n_queries, committee.score(iris['data'], iris['target'])))
#     plt.show()

##########################################################################################################################################################################
# import numpy as np  
# from sklearn.ensemble import IsolationForest  
# from sklearn.decomposition import PCA  
# from sklearn.cluster import DBSCAN  
# import pandas as pd  

# Project_folder = "D:\\keti\\zjh\\"   
# path = Project_folder + "modified_file1.xlsx"
#                                  # 设置空列表，用于存放样本的index
# df = pd.read_excel(path)
# # Step 1: 数据加载与预处理  
# # 假设 data 是一个 (1000, 100) 的 NumPy 数组  
# # data = np.random.rand(1000, 100)  # 示例数据 
# data=label = np.array(df.iloc[:, 17:115],dtype=np.float32)  
# data_normalized = (data - np.min(data, axis=1, keepdims=True)) / (np.max(data, axis=1, keepdims=True) - np.min(data, axis=1, keepdims=True))  

# # Step 2: 初步异常检测  
# # 使用孤立森林检测异常  
# isolation_forest = IsolationForest(contamination=0.05, random_state=42)  
# anomaly_scores = isolation_forest.fit_predict(data_normalized)  

# # 找出初步异常数据  
# anomalies = np.where(anomaly_scores == -1)[0]  
# print(f"初步检测到的异常数据索引: {anomalies}")  

# # Step 3: 主动学习策略  
# # 使用 PCA 降维以便可视化和进一步分析  
# pca = PCA(n_components=2)  
# data_pca = pca.fit_transform(data_normalized)  

# # 使用 DBSCAN 聚类，进一步筛选异常点  
# dbscan = DBSCAN(eps=0.5, min_samples=5)  
# clusters = dbscan.fit_predict(data_pca)  

# # 找出噪声点（DBSCAN 中的 -1 类）  
# noise_points = np.where(clusters == -1)[0]  
# print(f"DBSCAN 检测到的噪声点索引: {noise_points}")  

# # Step 4: 人工验证与迭代  
# # 将孤立森林和 DBSCAN 的结果结合，选择需要进一步分析的数据  
# potential_errors = set(anomalies).union(set(noise_points))  
# print(f"需要进一步分析的数据索引: {potential_errors}")
# a=list(potential_errors)
# for i in a:
#     print(data[i][0])

# output_file = "output.xlsx"  # 输出文件名  
# df_cleaned = df.drop(index=a)  

# # 将结果保存到新的 Excel 文件  
# df_cleaned.to_excel(output_file, index=False)  

# print(f"处理完成，剔除后的数据已保存到 {output_file}")
# import matplotlib.pyplot as plt  

# # 可视化降维后的数据  
# plt.scatter(data_pca[:, 0], data_pca[:, 1], c='blue', label='Normal Data')  
# plt.scatter(data_pca[list(potential_errors), 0], data_pca[list(potential_errors), 1], c='red', label='Potential Errors')  
# plt.legend()  
# plt.title("Data Visualization with Potential Errors Highlighted")  
# plt.show()
###############################################################################################################################################################################
import numpy as np  
import pandas as pd  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from skorch import NeuralNetRegressor  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from F_model_8paras import F_Net_2D
import joblib  

data = pd.read_excel(".\\train.xlsx")  

# 第 1-5 列作为输入特征，第 6-10 列作为目标值  
input1 = np.array(data.iloc[:, 1:4],dtype=np.float32)
input2 = np.array(data.iloc[:, 5:9],dtype=np.float32)
input3 = np.array(data.iloc[:, 10:14],dtype=np.float32)
input4 = np.array(data.iloc[:, 15],dtype=np.float32)
input4=np.expand_dims(input4, axis=0)
X=np.concatenate((input1, input2, input3,input4.reshape(input1.shape[0],1)),axis=1) 

y = data.iloc[:, 17:114].values.astype('float32')  # 目标值  

# 数据预处理：标准化  
scaler_X = StandardScaler()  
scaler_y = StandardScaler()  
X = scaler_X.fit_transform(X)  
y = scaler_y.fit_transform(y)  

# 将数据划分为训练池和未标注池  
X_train, X_pool, y_train, y_pool = train_test_split(X, y, test_size=0.2, random_state=42)  #x_train (2000,7) x_pool (500,7) y_train (2000,5) y_pool (500,5)

# 3. 定义主动学习的初始标注数据  
# 随机选择 10 个样本作为初始标注数据  
initial_indices = np.random.choice(range(len(X_train)), size=300, replace=False)  
X_initial = X_train[initial_indices]  
y_initial = y_train[initial_indices]  

# 剩余未标注数据  
X_train = np.delete(X_train, initial_indices, axis=0)  #(500,7)
y_train = np.delete(y_train, initial_indices, axis=0)  

# 4. 定义 NeuralNetRegressor  
net = NeuralNetRegressor(  
    F_Net_2D,  
    max_epochs=2000,  
    lr=0.001,  
    batch_size=100,  
    optimizer=torch.optim.Adam,  
    iterator_train__shuffle=True,  
    criterion=nn.MSELoss,  # 损失函数  
    verbose=0,  # 是否打印训练日志  
    device='cuda' 
    
    
)  
y_pool= scaler_y.inverse_transform(y_pool)  # 反标准化
# 5. 主动学习循环  
for iteration in range(10):  # 主动学习迭代 5 次  
    n=0
    print(f"Active Learning Iteration {iteration + 1}")  
    # 训练模型  
    net.fit(X_initial.reshape(-1, 1, 12), y_initial)  # 将数据调整为 (N, C, H, W) 格式  
    # 从未标注池中选择不确定性最高的样本  
    # 使用模型预测未标注数据  
    y_pred = net.predict(X_train.reshape(-1, 1, 12))  
    uncertainty = np.mean(np.abs(y_pred - y_train), axis=1)  # 计算预测值与真实值的差异（不确定性）  
    query_indices = np.argsort(uncertainty)[-20:]  # 选择不确定性最高的 10 个样本  
    
    # 将选中的样本加入标注数据  
    X_query = X_train[query_indices]  
    y_query = y_train[query_indices]  
    X_initial = np.vstack([X_initial, X_query])  
    y_initial = np.vstack([y_initial, y_query])  
    
    # 从未标注池中移除选中的样本  
    X_train = np.delete(X_train, query_indices, axis=0)  
    y_train = np.delete(y_train, query_indices, axis=0)  

    # 6. 测试模型  
    # 假设测试集为 X_pool  
    y_test_pred = net.predict(X_pool.reshape(-1, 1, 12))  
    y_test_pred = scaler_y.inverse_transform(y_test_pred)  # 反标准化 

    # print("Final Predictions:", y_test_pred[:5])
    for i in range(y_test_pred.shape[0]-1):
        for j in range(96):
            a=y_pool[i,j] 
            b=y_test_pred[i,j]
            if abs(b-a)<abs(0.05*a):
                n=n+1
    print(n/(97*y_test_pred.shape[0]))
joblib.dump(net, 'skorch_model.pkl') 

    