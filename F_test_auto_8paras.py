"""本代码用于从特定数据文件中随机抽取若干个样本用于测试对比并绘图"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from random import choice
import pandas as pd  
import random
Project_folder = ".\\"                 # 项目文件夹路径
# Project_folder = "G:\\SunHY\\Code\\[MLHB]\\"

"""定义超参数"""
# 如果网络能在GPU中训练，就使用GPU；否则使用CPU进行训练
DEVICE = torch.device('cpu')
# # 每批处理的数据
# BATCH_SIZE = 10000        # 一次跑完所有测试集
# # 训练轮次
# EPOCHS = 1                # 因为是测试，只需执行一轮

"""加载训练完成的模型，并在cpu中预测"""
model_pth = "F_train_5paras_d1_2000-[1112].pth"
net = torch.load(Project_folder  + model_pth, map_location=DEVICE)

# """训练范围外的数据测试"""
# path = "G:\\SunHY\\Code\\【Multi-layer Honeycomb CNN】\\【Data】\\Training Data\\"
# file_folder = 'Samples[30001-30010]'

"""训练范围内的数据测试"""
path = Project_folder + "val.xlsx"
                                 # 设置空列表，用于存放样本的index
df = pd.read_excel(path)
length=df.shape[0]    

def random_integers(k, a):  
    # 从 1 到 k 随机选择 a 个不重复的整数  
    if a > k:  
        raise ValueError("a cannot be greater than k")  
    
    random_numbers = random.sample(range(1, k + 1), a)  
    return random_numbers  

# 示例使用  
a = 20   # 设置要选择的整数个数  
result = random_integers(length-1, a)  
                                              # 要抽取的样本个数
for i in result:
    input1 = np.array(df.iloc[i, 1:4],dtype=np.float32)
    input2 = np.array(df.iloc[i, 5:9],dtype=np.float32)
    input3 = np.array(df.iloc[i, 10:14],dtype=np.float32)
    input4 = np.array(df.iloc[i, 15],dtype=np.float32)
    input4=np.expand_dims(input4, axis=0)
    input=np.concatenate((input1, input2, input3, input4))                                                # 读取结构参数
    label = np.array(df.iloc[i, 17:114],dtype=np.float32)    # 读取351个频点对应的RL
    freq = np.arange(1.025, 17.875 , 0.175)  # 加上 0.175 是为了确保包含 17.875 

    input = torch.tensor(input)                             # 将ndarry转换为tensor
    input = torch.unsqueeze(input, dim=0)                   # input.shape从[2]→[1, 2]
    input = torch.unsqueeze(input, dim=0)                   # 将input.shape从[1, 2]→[1, 1, 2]，才可作为网络输入用于预测

    pred = net(input)                                       # 将input输入训练完成的网络，得到预测值
    pred = pred.detach().numpy()                            # 将pred去除梯度，再转为numpy格式
    pred = np.squeeze(pred)                                 # 去除为1的维度，将预测值转换为一维数组，用于作图


    # 预测曲线作图
    x_Freq = freq                           # 横坐标 Frequency/GHz
    y_Pred = pred                           # 预测曲线 S11
    y_Label = label                         # CST仿真曲线 S11

    # 设置图表的标题，横坐标与纵坐标的名称
    # plt.title("CNN prediction compared to CST simulation\n" + "From " + file_folder + '_' + file_name)
    plt.xlabel("Freq / GHz", fontsize=12)
    plt.ylabel("RL / dB", fontsize=12)

    plt.tick_params(labelsize=12)
    plt.plot(x_Freq, y_Pred, label='CNN', color='k', linewidth=1.5)       # 画CNN网络预测的曲线
    plt.plot(x_Freq, y_Label, label='feko', color='k', linewidth=2, linestyle='--')      # 画CST仿真得到的曲线
    plt.xlim(0, 18.5)
    plt.legend(loc="best", fontsize=12)                      # 自动选择最合适位置显示图例
    plt.show()                                  # 显示对比曲线
