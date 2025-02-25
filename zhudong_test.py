
import joblib  
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  
import random
data = pd.read_excel("val.xlsx")  

# 第 1-5 列作为输入特征，第 6-10 列作为目标值  
input1 = np.array(data.iloc[:, 1:4],dtype=np.float32)
input2 = np.array(data.iloc[:, 5:9],dtype=np.float32)
input3 = np.array(data.iloc[:, 10:14],dtype=np.float32)
input4 = np.array(data.iloc[:, 15],dtype=np.float32)
input4=np.expand_dims(input4, axis=0)
X=np.concatenate((input1, input2, input3,input4.reshape(input1.shape[0],1)),axis=1) 

y = data.iloc[:, 17:114].values.astype('float32')  # 目标值  

scaler_X = StandardScaler()  
scaler_y = StandardScaler()  
X = scaler_X.fit_transform(X)  
y1 = scaler_y.fit_transform(y)  
# 数据预处理：标准化  
# scaler_X = StandardScaler()  
# scaler_y = StandardScaler()  
# X = scaler_X.fit_transform(X)  
# y = scaler_y.fit_transform(y)  


# 加载模型  
loaded_net = joblib.load('skorch_model.pkl')  
print("模型已加载")  
freq = np.arange(1.025, 17.875 , 0.175)  # 加上 0.175 是为了确保包含 17.875 
y_pred = loaded_net.predict(X.reshape(-1,1,12))  
y_pred = scaler_y.inverse_transform(y_pred)  # 反标准化
# 使用加载的模型进行预测 
length=data.shape[0]    

def random_integers(k, a):  
    # 从 1 到 k 随机选择 a 个不重复的整数  
    if a > k:  
        raise ValueError("a cannot be greater than k")  
    
    random_numbers = random.sample(range(1, k + 1), a)  
    return random_numbers  
a = 10   # 设置要选择的整数个数  
result = random_integers(length-1, a)   
for i in result:
    
    # print(y_pred)
        # 预测曲线作图
    x_Freq = freq                           # 横坐标 Frequency/GHz
    y_Pred_s = y_pred[i]                           # 预测曲线 S11
    y_Label = y[i]                         # CST仿真曲线 S11

    # 设置图表的标题，横坐标与纵坐标的名称
    # plt.title("CNN prediction compared to CST simulation\n" + "From " + file_folder + '_' + file_name)
    plt.xlabel("Freq / GHz", fontsize=12)
    plt.ylabel("RL / dB", fontsize=12)

    plt.tick_params(labelsize=12)
    plt.plot(x_Freq, y_Pred_s, label='act', color='k', linewidth=1.5)       # 画CNN网络预测的曲线
    plt.plot(x_Freq, y_Label, label='feko', color='k', linewidth=2, linestyle='--')      # 画CST仿真得到的曲线
    plt.xlim(0, 18.5)
    plt.legend(loc="best", fontsize=12)                      # 自动选择最合适位置显示图例
    plt.show()      