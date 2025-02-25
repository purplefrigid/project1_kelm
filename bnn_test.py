import pyro  
import torch  
from F_model_8paras import F_Net_2D
from torch.utils.data import DataLoader, Dataset  
from F_Dataset_8paras import MH_Data
import pyro.poutine as poutine
import pyro.distributions as dist  
import matplotlib.pyplot as plt
from random import choice
import pandas as pd  
import random
import numpy as np
from scipy.spatial.distance import euclidean  
from scipy.stats import pearsonr  
from scipy.spatial.distance import cosine  
from fastdtw import fastdtw  
# 假设 `guide` 是你的导向模型  
# 提取 guide 中的参数  
Project_folder = ".\\" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
mh_val = MH_Data(Project_folder + 'test.xlsx')

val_loader = DataLoader(batch_size=10, dataset=mh_val, shuffle=True,
                            num_workers=1, pin_memory=True, persistent_workers=True)
net =F_Net_2D().to(device)
pyro.get_param_store().load("guide_params.pt")  

# 打印所有参数  
param_store = pyro.get_param_store()  
priors = {}  
def random_integers(k, a):  
    # 从 1 到 k 随机选择 a 个不重复的整数  
    if a > k:  
        raise ValueError("a cannot be greater than k")  
    
    random_numbers = random.sample(range(1, k + 1), a)  
    return random_numbers  

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
                similarity_matrix[i][j] = min(similarity_matrix[i][j], cosine_similarity)  

                # 3. 计算皮尔逊相关系数  
                pearson_corr, _ = pearsonr(curves[i], curves[j])  
                similarity_matrix[i][j] = min(similarity_matrix[i][j], pearson_corr)  

                # 4. 动态时间规整（DTW）  
                dtw_distance, _ = fastdtw(curves[i], curves[j])  
                dtw_similarity = 1 / (1 + dtw_distance)  
                similarity_matrix[i][j] = min(similarity_matrix[i][j], dtw_similarity) 

                upper_triangle_indices = np.triu_indices(len(curves), k=1)  
                average_similarity = np.mean(similarity_matrix[upper_triangle_indices])   

    return average_similarity  
if __name__ == '__main__':
    
    for name_n, param in net.named_parameters():
        k=0
        for name, value in param_store.items():  
        # 将参数名称作为字典的键，参数值作为字典的值  
            if name == name_n+"_loc":
                loc = value
                k=1
            if name == name_n+"_scale" :
                scale = value
                k=2
            if k==2 :
                priors[name_n] = dist.Normal(loc, scale).to_event(param.dim())


        # for name_n, param in net.named_parameters():
        #     for name_b, parab in bnn.named_parameters():
        #         if name_b == name_n:
        #             print("name_n:",name_n)
        #             print("name_b:",name_b)
        #             print("param_n:",param)
        #             print("param_b:",parab)
        # for x, y in val_loader:
        #     x=x.to(device) 
        #     y=y.to(device)  
        #     print(x.shape)
        #     output = bnn(x)
        #     output2 = net(x)  
        #     print("Model output:", output)  
        #     # print("real output:", y)
        path = Project_folder + "val.xlsx"
                                    # 设置空列表，用于存放样本的index
        df = pd.read_excel(path)
        length=df.shape[0]    
        a = 10   # 设置要选择的整数个数  
        result = random_integers(length-1, a)  
        # result=[9]
                                                # 要抽取的样本个数
    for i in result:
        curves=[]

        input1 = np.array(df.iloc[i, 1:4],dtype=np.float32)
        input2 = np.array(df.iloc[i, 5:9],dtype=np.float32)
        input3 = np.array(df.iloc[i, 10:14],dtype=np.float32)
        input4 = np.array(df.iloc[i, 15],dtype=np.float32)
        input4=np.expand_dims(input4, axis=0)
        input=np.concatenate((input1, input2, input3, input4))                                                # 读取结构参数
        label = np.array(df.iloc[i, 17:114],dtype=np.float32)    # 读取351个频点对应的RL
        freq = np.arange(1.025, 17.875 , 0.175)  # 加上 0.175 是为了确保包含 17.875 
        color=['r','g','b','c','m','y']
        input = torch.tensor(input)  .to(device)                           # 将ndarry转换为tensor
        input = torch.unsqueeze(input, dim=0)                   # input.shape从[2]→[1, 2]
        input = torch.unsqueeze(input, dim=0)                   # 将input.shape从[1, 2]→[1, 1, 2]，才可作为网络输入用于预测
        # def define_variables(n):  
# for i in range(1, n + 1):  
# globals()[f'var{i}'] = 1  # 动态创建变量  

# # 定义变量数量  
# n = 3  
# define_variables(n)  

# # 循环打印每个动态变量的值  
# for i in range(1, n + 1):  
# print(f'var{i} = {globals()[f"var{i}"]}')  # 使用 globals() 访问动态变量
        num=4
        x_Freq = freq                           # 横坐标 Frequency/GHz
        # y_Pred = pred                           # 预测曲线 S11
        y_Label = label                         # CST仿真曲线 S11
        for j in range(num):
            lifted_bnn = pyro.random_module("bnn", net, priors)
            bnn = lifted_bnn()
            # if j ==0:
            #     pred = bnn(input)   
            #     pred=pred.to("cpu")                                    # 将input输入训练完成的网络，得到预测值
            #     pred = pred.detach().numpy()                            # 将pred去除梯度，再转为numpy格式
            #     pred = np.squeeze(pred)                                 # 去除为1的维度，将预测值转换为一维数组，用于作图
            # if j==1:
            #     pred1 = bnn(input)   
            #     pred1=pred1.to("cpu")                                    # 将input输入训练完成的网络，得到预测值
            #     pred1 = pred1.detach().numpy()                            # 将pred去除梯度，再转为numpy格式
            #     pred1 = np.squeeze(pred1)  
            # if j==2:
            #     pred2 = bnn(input)   
            #     pred2=pred2.to("cpu")                                    # 将input输入训练完成的网络，得到预测值
            #     pred2 = pred2.detach().numpy()                            # 将pred去除梯度，再转为numpy格式
            #     pred2 = np.squeeze(pred2)  
            globals()[f'var{j}'] = bnn(input)   
            globals()[f'var{j}']=globals()[f'var{j}'].to("cpu")                                    # 将input输入训练完成的网络，得到预测值
            globals()[f'var{j}'] = globals()[f'var{j}'].detach().numpy()                            # 将pred去除梯度，再转为numpy格式
            globals()[f'var{j}'] = np.squeeze(globals()[f'var{j}'])    
            curves.append(globals()[f'var{j}'])
            plt.plot(x_Freq, globals()[f'var{j}'], label='bnn'+str(j), color=color[j], linewidth=1.5)       # 画CNN网络预测的曲线
        average_similarity1=calculate_similarity(curves)              
        mean_vector = np.mean(curves, axis=0) 
        plt.plot(x_Freq, mean_vector, label='mean', color='k', linewidth=1.5,linestyle=':')  
        curves.clear()  
        curves.append(mean_vector)  
        curves.append(y_Label)  
        average_similarity2=calculate_similarity(curves)
    # 预测曲线作图


        # 设置图表的标题，横坐标与纵坐标的名称
        plt.title(f'Predict Similarity: {average_similarity1:.4f}\nOverall Similarity: {average_similarity2:.4f}')
        plt.xlabel("Freq / GHz", fontsize=12)
        plt.ylabel("RL / dB", fontsize=12)

        plt.tick_params(labelsize=12)
        # plt.plot(x_Freq, pre, label='bnn1', color='r', linewidth=1.5)       # 画CNN网络预测的曲线
        # plt.plot(x_Freq, pred1, label='bnn2', color='g', linewidth=1.5)       # 画CNN网络预测的曲线
        # plt.plot(x_Freq, pred2, label='bnn2', color='b', linewidth=1.5)       # 画CNN网络预测的曲线
        plt.plot(x_Freq, y_Label, label='feko', color='k', linewidth=2, linestyle='--')      # 画CST仿真得到的曲线
        plt.xlim(0, 18.5)
        plt.legend(loc="best", fontsize=12)                      # 自动选择最合适位置显示图例
        plt.show()                                  # 显示对比曲线