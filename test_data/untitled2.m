%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  导入数据
data = xlsread('data_90.xlsx');
data1 = xlsread("data_0.xlsx");
%%  划分训练集和测试集
x = data(1:end,1);
y1 = data(1:end,2);
y2 = data(1:end,3);

x1 = data1(1:end,1);
y11 = data1(1:end,2);
y21 = data1(1:end,3);
%%  绘图
figure
plot(x, y1, 'r-*',x, y2, 'b-o', 'LineWidth', 1)
xlabel('频率')
ylabel('RCS')
xlim([7, 13])
ylim([-35, -10])
grid

figure
plot(x1, y11, 'r-*',x1, y21, 'b-o', 'LineWidth', 1)
xlabel('频率')
ylabel('RCS')
xlim([7, 13])
ylim([-35, -10])
grid