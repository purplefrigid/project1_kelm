import os
import random

import torch
import time
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from F_Dataset_8paras import MH_Data
from F_model_8paras import F_Net_2D,F_Net_3D
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


Project_folder = ".\\" 
#Project_folder = "O:\\Python Files\\[MLHB]\\"                   # 项目文件夹路径
# Project_folder = "G:\\SunHY\\Code\\[MLHB]\\"


if __name__ == '__main__':

    # 锁定训练随机种子
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = True

    seed = 1112
    setup_seed(seed)

    """获取训练代码文件名与开始时间"""
    # 获取训练代码的文件名
    py_name = (os.path.basename(__file__)).split('.')[0]    # .split('.')[0] 删除含扩展名的文件名的从’.‘往后的内容
    # 打印训练开始时间
    print(py_name + "\n网络训练开始时间:", time.strftime('%Y-%m-%d %H:%M:%S\n', time.localtime(time.time())))

    """定义超参数"""
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       # 网络训练设备，默认GPU
    BATCH_SIZE = 200                                                            # 每批处理的数据
    EPOCHS = 50                                                           # 定义训练的最大轮次
    LEARNING_RATE = 0.001
    print(DEVICE)
    # model_pth = "F_train_5paras2_2500-[1112].pth"
    # net = torch.load(Project_folder  + model_pth, map_location=DEVICE)
    """网络训练数据导入"""
    # 基于MH_Data类，创建训练与验证数据集
    # path = Project_folder + 'samples\\'
    mh_train = MH_Data(Project_folder + 'train.xlsx')
    mh_val = MH_Data(Project_folder + 'test.xlsx')
    # path = Project_folder + 'Data\\shy_8paras\\'
    # mh_train = MH_Data(path + 'Sample_S_50000_pp\\train')
    # mh_val = MH_Data(path + 'Sample_S_50000_pp\\val')

    # 载入训练集与验证集
    train_loader = DataLoader(batch_size=BATCH_SIZE, dataset=mh_train, shuffle=True, drop_last=True,
                              num_workers=1, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(batch_size=20, dataset=mh_val, shuffle=True,
                            num_workers=1, pin_memory=True, persistent_workers=True)

    """网络框架搭建"""
    # 导入网络模型
    net = F_Net_2D()
    # net2 = F_Net_3D().to(   DEVICE  )
    # model_pth = "F_train_5paras3_2000-[1112].pth"
    # net = torch.load(Project_folder  + model_pth, map_location=DEVICE)
    # net = torch.load('G:\\SunHY\\Code\\【Multi-layer Honeycomb CNN】\\【Net】\\Forward Net 1D\\F_train_1D_1000-[1234].pth')
    # net.apply(init_weight)

    # 将模型传入到device中，并将其结构显示出来
    print(net.to(DEVICE))

    # 定义损失函数与优化器
    lossF = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    # optimizer2 = torch.optim.AdamW(net2.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    print('\nBatch_Size=' + str(BATCH_SIZE), '\t\tLearning_Rate='+str(LEARNING_RATE), '\t\tRandom_Seed=' + str(seed))

    """网络训练过程"""
    # 存储训练过程
    train_history = {'train_AvgLoss': [], 'train_AvgAccuracy': []}
    # train2_history = {'train2_AvgLoss': [], 'train2_AvgAccuracy': []}
    val_history = {'val_AvgLoss': [], 'val_AvgAccuracy': []}
    total_time = 0

    # 开始网络训练
    for epoch in range(1, EPOCHS + 1):
        net.train(True)  # 打开网络的训练模式
        train_start = time.time()                   # 记录本轮训练开始时间
        t_totalLoss = 0                             # 初始化训练集总损失函数
        t_totalAcc = 0                              # 初始化训练集总准确率

        for step, (t_inputs, t_labels) in enumerate(train_loader):
            t_inputs = t_inputs.to(DEVICE)          # 将结构参数和S11标签传输进device中
            t_labels = t_labels.to(DEVICE)
            # print(t_inputs.shape)
            # print(t_labels.shape)
            t_labels = torch.squeeze(t_labels)      # 删除labels中维度为1的维度，使其shape从[bs, 1, 351]变为[bs, 351]，
            # 此步是为了确保labels维度与outputs相同，才可以计算loss
            # print(t_inputs.shape)
            # print(t_labels.shape)
            optimizer.zero_grad()                   # 清空模型的梯度
            t_outputs = net(t_inputs)               # 对模型进行前向推理

            t_outputs = torch.squeeze(t_outputs)

            # 计算损失与反向传播
            loss = lossF(t_outputs, t_labels)                   # 计算本轮推理的Loss值
            loss.backward()                                     # 进行反向传播求出模型参数的梯度

            t_totalLoss += loss                                 # 计算损失函数总和
            # train_AvgLoss = t_totalLoss / len(train_loader)     # 计算平均损失函数

            # 获取训练时的准确率
            outlist = t_outputs.view(-1).tolist()           # 将输出的tensor[bs, 351]展平为tensor[bs*351]，再转换为列表
            lablist = t_labels.view(-1).tolist()            # 将标签的tensor[bs, 6]展平为tensor[bs*6]，再转换为列表

            count = 0                                       # 初始化计数变量
            for out, lab in zip(outlist, lablist):          # 同时遍历outlist和lablist两个列表
                if abs(out - lab) < abs(lab) * 0.05:        # 当out与lab的差值小于0.05*lab时，则视为准确
                    count += 1  # 同时计数＋1
            train_acc = count / (len(lablist))                # 计数除以列表长度，即可算出准确率
            t_totalAcc += train_acc

            optimizer.step()       
            # train_stop = time.time()                        # 记录本轮训练结束时间
            # train_time = train_stop - train_start           # 计算本轮训练时间
            # total_time += train_time / 60                   # 获取网络训练累计时间                             # 使用迭代器更新模型权重
            # print('\n本轮训练用时' + format(train_time, '.2f') + 's，总用时' + format(total_time, '.2f') + 'min.')
            if step == len(train_loader) - 1:                   # 当一轮epoch运行到最后一个step时，
                v_totalLoss = 0                                 # 构造临时变量
                v_totalAcc = 0
                net.train(False)                                # 关闭模型的训练状态

                with torch.no_grad():                           # 不计算梯度
                    for v_inputs, v_labels in val_loader:       # 对验证集进行迭代
                        v_inputs = v_inputs.to(DEVICE)
                        v_labels = v_labels.to(DEVICE)
                        v_labels = torch.squeeze(v_labels)
                        v_outputs = net(v_inputs)

                        loss = lossF(v_outputs, v_labels)
                        v_totalLoss += loss                           # 存储测试结果

                        # 获取验证集的准确率
                        outlist = v_outputs.view(-1).tolist()       # 将输出的tensor[16, 321]展平为tensor[5136]，
                        lablist = v_labels.view(-1).tolist()        # 再转换为列表用于遍历
                        count = 0  # 构造计数变量
                        for out, lab in zip(outlist, lablist):      # 同时遍历outlist和lablist两个列表
                            if abs(out - lab) < abs(lab) * 0.05:    # 当out与lab的差值小于0.05*lab时，则视为准确
                                count += 1                          # 同时计数＋1
                        v_accuracy = count / len(lablist)           # 计数除以列表长度，即可算出准确率

                        v_totalAcc += v_accuracy                    # 将每一组验证集的准确率相加

                    val_AvgLoss = v_totalLoss / len(val_loader)           # 计算验证集的平均Loss
                    val_AvgAcc = v_totalAcc / len(val_loader)        # 计算验证集的平均准确率
                    val_history['val_AvgLoss'].append(val_AvgLoss.item())
                    val_history['val_AvgAccuracy'].append(val_AvgAcc)

                train_AvgLoss = t_totalLoss / len(train_loader)
                train_AvgAcc = t_totalAcc / len(train_loader)
                train_history['train_AvgLoss'].append(train_AvgLoss.item())
                train_history['train_AvgAccuracy'].append(train_AvgAcc)

                train_stop = time.time()                        # 记录本轮训练结束时间
                train_time = train_stop - train_start           # 计算本轮训练时间
                total_time += train_time / 60                   # 获取网络训练累计时间

                # 将本step结果进行可视化处理
                print('\n本轮训练用时' + format(train_time, '.2f') + 's，总用时' + format(total_time, '.2f') + 'min.')
                print(" [%d/%d] train_AvgLoss: %.4f, train_AvgAccuracy: %.4f,"
                        " val_AvgLoss: %.4f, val_AvgAccuracy: %.4f  " %
                        (epoch, EPOCHS, train_AvgLoss.item(), train_AvgAcc, val_AvgLoss.item(), val_AvgAcc))
                # print(" [%d/%d] train_AvgLoss: %.4f, train_AvgAccuracy: %.4f,"
                #         %
                #         (epoch, EPOCHS, train_AvgLoss.item(), train_AvgAcc))
    # for epoch in range(1, EPOCHS + 1):
    #     net2.train(True)  # 打开网络的训练模式
    #     train_start = time.time()                   # 记录本轮训练开始时间
    #     t_totalLoss = 0                             # 初始化训练集总损失函数
    #     t_totalAcc = 0                              # 初始化训练集总准确率

    #     for step, (t_inputs, t_labels) in enumerate(train_loader):
    #         t_inputs = t_inputs.to(DEVICE)          # 将结构参数和S11标签传输进device中
    #         t_labels = t_labels.to(DEVICE)
    #         # print(t_inputs.shape)
    #         # print(t_labels.shape)
    #         t_labels = torch.squeeze(t_labels)      # 删除labels中维度为1的维度，使其shape从[bs, 1, 351]变为[bs, 351]，
    #         # 此步是为了确保labels维度与outputs相同，才可以计算loss
    #         # print(t_inputs.shape)
    #         # print(t_labels.shape)
    #         optimizer2.zero_grad()                   # 清空模型的梯度
    #         t_outputs = net2(t_inputs)               # 对模型进行前向推理

    #         t_outputs = torch.squeeze(t_outputs)

    #         # 计算损失与反向传播
    #         loss = lossF(t_outputs, t_labels)                   # 计算本轮推理的Loss值
    #         loss.backward()                                     # 进行反向传播求出模型参数的梯度

    #         t_totalLoss += loss                                 # 计算损失函数总和
    #         # train_AvgLoss = t_totalLoss / len(train_loader)     # 计算平均损失函数

    #         # 获取训练时的准确率
    #         outlist = t_outputs.view(-1).tolist()           # 将输出的tensor[bs, 351]展平为tensor[bs*351]，再转换为列表
    #         lablist = t_labels.view(-1).tolist()            # 将标签的tensor[bs, 6]展平为tensor[bs*6]，再转换为列表

    #         count = 0                                       # 初始化计数变量
    #         for out, lab in zip(outlist, lablist):          # 同时遍历outlist和lablist两个列表
    #             if abs(out - lab) < abs(lab) * 0.05:        # 当out与lab的差值小于0.05*lab时，则视为准确
    #                 count += 1  # 同时计数＋1
    #         train_acc = count / (len(lablist))                # 计数除以列表长度，即可算出准确率
    #         t_totalAcc += train_acc

    #         optimizer2.step()       
    #         # train_stop = time.time()                        # 记录本轮训练结束时间
    #         # train_time = train_stop - train_start           # 计算本轮训练时间
    #         # total_time += train_time / 60                   # 获取网络训练累计时间                             # 使用迭代器更新模型权重
    #         # print('\n本轮训练用时' + format(train_time, '.2f') + 's，总用时' + format(total_time, '.2f') + 'min.')
    #         if step == len(train_loader) - 1:                   # 当一轮epoch运行到最后一个step时，
    #             v_totalLoss = 0                                 # 构造临时变量
    #             v_totalAcc = 0
    #             net2.train(False)                                # 关闭模型的训练状态

    #             # with torch.no_grad():                           # 不计算梯度
    #             #     for v_inputs, v_labels in val_loader:       # 对验证集进行迭代
    #             #         v_inputs = v_inputs.to(DEVICE)
    #             #         v_labels = v_labels.to(DEVICE)
    #             #         v_labels = torch.squeeze(v_labels)
    #             #         v_outputs = net(v_inputs)

    #             #         loss = lossF(v_outputs, v_labels)
    #             #         v_totalLoss += loss                           # 存储测试结果

    #             #         # 获取验证集的准确率
    #             #         outlist = v_outputs.view(-1).tolist()       # 将输出的tensor[16, 321]展平为tensor[5136]，
    #             #         lablist = v_labels.view(-1).tolist()        # 再转换为列表用于遍历
    #             #         count = 0  # 构造计数变量
    #             #         for out, lab in zip(outlist, lablist):      # 同时遍历outlist和lablist两个列表
    #             #             if abs(out - lab) < abs(lab) * 0.05:    # 当out与lab的差值小于0.05*lab时，则视为准确
    #             #                 count += 1                          # 同时计数＋1
    #             #         v_accuracy = count / len(lablist)           # 计数除以列表长度，即可算出准确率

    #             #         v_totalAcc += v_accuracy                    # 将每一组验证集的准确率相加

    #             #     val_AvgLoss = v_totalLoss / len(val_loader)           # 计算验证集的平均Loss
    #             #     val_AvgAcc = v_totalAcc / len(val_loader)        # 计算验证集的平均准确率
    #             #     val_history['val_AvgLoss'].append(val_AvgLoss.item())
    #             #     val_history['val_AvgAccuracy'].append(val_AvgAcc)

    #             train_AvgLoss = t_totalLoss / len(train_loader)
    #             train_AvgAcc = t_totalAcc / len(train_loader)
    #             train2_history['train2_AvgLoss'].append(train_AvgLoss.item())
    #             train2_history['train2_AvgAccuracy'].append(train_AvgAcc)

    #             train_stop = time.time()                        # 记录本轮训练结束时间
    #             train_time = train_stop - train_start           # 计算本轮训练时间
    #             total_time += train_time / 60                   # 获取网络训练累计时间

    #             # 将本step结果进行可视化处理
    #             print('\n本轮训练用时' + format(train_time, '.2f') + 's，总用时' + format(total_time, '.2f') + 'min.')
    #             # print(" [%d/%d] train_AvgLoss: %.4f, train_AvgAccuracy: %.4f,"
    #             #         " val_AvgLoss: %.4f, val_AvgAccuracy: %.4f  " %
    #             #         (epoch, EPOCHS, train_AvgLoss.item(), train_AvgAcc, val_AvgLoss.item(), val_AvgAcc))
    #             print(" [%d/%d] train_AvgLoss: %.4f, train_AvgAccuracy: %.4f,"
    #                     %
    #                     (epoch, EPOCHS, train_AvgLoss.item(), train_AvgAcc))
        # 中途停止网络训练的指标
        # if epoch >= 300 and train_AvgAcc >= 0.99 and val_AvgAcc >= 0.925:
        #     EPOCHS = epoch
        #     break


    # 将history转换为dataframe，并保存为txt
    Log = pd.DataFrame(data=[train_history['train_AvgLoss'], val_history['val_AvgLoss'],
                             train_history['train_AvgAccuracy'], val_history['val_AvgAccuracy']],
                       index=['train_AvgLoss', 'val_AvgLoss', 'train_AvgAccuracy', 'val_AvgAccuracy'])
    # Log = pd.DataFrame(data=[train_history['train_AvgLoss'], train2_history['train2_AvgLoss'],
    #                          train_history['train_AvgAccuracy'], train2_history['train2_AvgAccuracy']],
    #                    index=['train_AvgLoss', 'train2_AvgLoss', 'train_AvgAccuracy', 'train2_AvgAccuracy'])
    Log.columns = Log.columns + 1
    Log_file = Project_folder + "Net\\shy_8paras\\ForwardNet\\" + py_name + "_" + str(EPOCHS) + ".txt"
    Log.to_csv(Log_file, sep='\t')
    # print(train_history, val_history)

    x_epochs = list(range(1, EPOCHS+1))         # 获取横坐标Epoch

    """利用matplotlib.pyplot可视化训练过程"""
    # 打印损失函数变化曲线
    plt.title("Average Loss from " + py_name)                           # 在绘制图表的标题中体现出训练代码的文件名
    # plt.plot(x_epochs, train2_history['train2_AvgLoss'], label='act_AvgLoss')
    plt.plot(x_epochs, train_history['train_AvgLoss'], label='train_AvgLoss')
    plt.plot(x_epochs, val_history['val_AvgLoss'], label='val_AvgLoss')
    plt.legend(loc='best')                                              # 自动选择最合适位置显示图例
    plt.grid(False)                                                     # 不显示网格线
    plt.xlabel('Epoch')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))        # 将x轴坐标设置为整数
    plt.ylabel('Loss')
    plt.show()

    # 打印准确率变化曲线
    plt.title("Average Accuracy from " + py_name)
    # plt.plot(x_epochs, train2_history['train2_AvgAccuracy'], label='act_AvgAccuracy')
    plt.plot(x_epochs, train_history['train_AvgAccuracy'], label='train_AvgAccuracy')
    plt.plot(x_epochs, val_history['val_AvgAccuracy'], label='val_AvgAccuracy')
    plt.legend(loc='best')
    plt.grid(False)
    plt.xlabel('Epoch')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel('Accuracy')
    plt.show()

    print("\n网络训练结束时间:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))   # 打印当前时间

    torch.save(net, '.\\' + py_name+'_d2' + '_' + str(EPOCHS) + '-[' + str(seed) + '].pth')     # 将网络保存为于.py同名的.pth文件exit





