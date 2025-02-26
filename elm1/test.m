%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  导入数据
res = xlsread('combined_output.xlsx');

%%  划分训练集和测试集
temp = randperm(300);

P_train = res(temp(1: 100), 1: 8)';
T_train = res(temp(1: 100), 9:end)';
M = size(P_train, 2);

P_test = res(temp(101: 200), 1: 8)';
T_test = res(temp(101: 200), 9:end)';
N = size(P_test, 2);

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  创建模型
num_hiddens = 50;        % 隐藏层节点个数
activate_model = 'sig';  % 激活函数
[IW, B, LW, TF, TYPE] = elmtrain(p_train, t_train, num_hiddens, activate_model, 0);

%%  仿真测试
t_sim1 = elmpredict(p_train, IW, B, LW, TF, TYPE);
t_sim2 = elmpredict(p_test , IW, B, LW, TF, TYPE);

%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%%  均方根误差
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);

%%  绘图
figure
plot(1: 351, T_train(:,1), 'r-*', 1: 351, T_sim1(:,1), 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1, 351])
grid

figure
plot(1: 351, T_test(:,1), 'r-*', 1: 351, T_sim2(:,1), 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['RMSE=' num2str(error2)]};
title(string)
xlim([1, 351])
grid

%%  相关指标计算
% R2
% R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
% R2 = 1 - norm(T_test  - T_sim2)^2 / norm(T_test  - mean(T_test ))^2;
% 
% disp(['训练集数据的R2为：', num2str(R1)])
% disp(['测试集数据的R2为：', num2str(R2)])
% 
% % MAE
% mae1 = sum(abs(T_sim1 - T_train)) ./ M ;
% mae2 = sum(abs(T_sim2 - T_test )) ./ N ;
% 
% disp(['训练集数据的MAE为：', num2str(mae1)])
% disp(['测试集数据的MAE为：', num2str(mae2)])
% 
% % MBE
% mbe1 = sum(T_sim1 - T_train) ./ M ;
% mbe2 = sum(T_sim2 - T_test ) ./ N ;
% 
% disp(['训练集数据的MBE为：', num2str(mbe1)])
% disp(['测试集数据的MBE为：', num2str(mbe2)])
% 
% %%  绘制散点图
% sz = 25;
% c = 'b';
% 
% figure
% scatter(T_train, T_sim1, sz, c)
% hold on
% plot(xlim, ylim, '--k')
% xlabel('训练集真实值');
% ylabel('训练集预测值');
% xlim([min(T_train) max(T_train)])
% ylim([min(T_sim1) max(T_sim1)])
% title('训练集预测值 vs. 训练集真实值')
% 
% figure
% scatter(T_test, T_sim2, sz, c)
% hold on
% plot(xlim, ylim, '--k')
% xlabel('测试集真实值');
% ylabel('测试集预测值');
% xlim([min(T_test) max(T_test)])
% ylim([min(T_sim2) max(T_sim2)])
% title('测试集预测值 vs. 测试集真实值')