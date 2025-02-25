"""本代码用于自定义数据集的编写"""

import numpy as np
import torch
import os

import pandas as pd  
from torch.utils.data import Dataset

# Project_folder = "C:\\Users\\zgnhz\\Desktop\\try\\"
#Project_folder = "O:\\Python Files\\[MLHB]\\"                   # 项目文件夹路径
# Project_folder = "G:\\SunHY\\Code\\[MLHB]\\"

# path = Project_folder + 'train.xlsx'  # sample文件保存路径
path =  './test.xlsx'
# path = Project_folder + "Data\\shy_8paras\\Sample_S_50000\\"  # sample文件保存路径
#path = os.path.join(Project_folder, "Data\\shy_8paras\\Sample_S_50000\\")  # sample文件保存路径
"""python中文件路径不能以'\'结尾，在文件路径前加r的方法在某些情况下可能不通用"""
# print(os.getcwd())                     # 当前工作目录【代码所在目录】
# os.chdir(path)                         # 修改工作目录至path路径

# 继承Dataset类创建子类
class MH_Data(Dataset):
    def __init__(self, file_path):                      # __init__()方法: 添加MH_Data类的属性【文件路径】
        self.file_path = file_path                          # 样本路径
        self.df = pd.read_excel(self.file_path)
        # self.sample_list = os.listdir(self.file_path)       # 样本路径下的文件列表
        # print("hellow")
    def __len__(self):                                  # __len__()方法: 读取文件列表的长度，即样本数量
        return(self.df.shape[0])

    def __getitem__(self, index):                       # __getitem__()方法: 从单个文件中提取训练集与数据集，可用于后续迭代
        # file = self.sample_list[index]                      # [index]是文件的索引，表示文件列表中的第i个文件
        # 通过numpy读取文件内容，x, y分别表示输入和标签

        # sample_raw = np.loadtxt(path+file, dtype=np.float32, delimiter='\t')
        # sample_raw = np.loadtxt(os.path.join(path, file), dtype=np.float32, delimiter='\t')
        input1 = np.array(self.df.iloc[index, 1:4],dtype=np.float32)
        input2 = np.array(self.df.iloc[index, 5:9],dtype=np.float32)
        input3 = np.array(self.df.iloc[index, 10:14],dtype=np.float32)
        input4 = np.array(self.df.iloc[index, 15],dtype=np.float32)
        input4=np.expand_dims(input4, axis=0)
        input=np.concatenate((input1, input2, input3, input4)) 
        label = np.array(self.df.iloc[index, 17:114],dtype=np.float32)
        input = torch.tensor(input)
        label = torch.tensor(label)

        """{【重要！！！！！】}需要给输入数据增加一个维度，将数据的tensor形式从[Length]转变为[Channel, Length]"""
        input = torch.unsqueeze(input, dim=0)
        label = torch.unsqueeze(label, dim=0)
        # print(input, input.shape, input.dtype)
        # print(label, label.shape, label.dtype)

        return input, label             # 返回输入与标签


# # 基于MH_Data类，创建mhdata数据集
# mhdata = MH_Data(path)

# """检查代码用"""
# print(mhdata.file_path)
# # print(mhdata.sample_list)
# print(mhdata.__len__())
# input, label=mhdata.__getitem__(10)
# print(input)
# print(label)
