%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

res = xlsread('keti.xlsx');

%  划分训练集和测试集
temp = randperm(87);
P_train = res(temp(1: 67), 1: 12)';
T_train = res(temp(1: 67), 13:end)';
M = size(P_train, 2);
P_test = res(temp(68: 87), 1: 12)';
T_test = res(temp(68: 87), 13:end)';
N = size(P_test, 2);
n=size(T_test,1);
%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

% trialNum=100;
% trainAccuary=zeros(trialNum,1);
% testAccuary=zeros(trialNum,1);
% TrainingTime=zeros(trialNum,1);
% TestingTime=zeros(trialNum,1);
%-------------------------------------------------------------------------
TrainingLData=[t_train',p_train'];
TestingData=[t_test',p_test'];

%保持随机过程不变
% rand('state',1);


%for ii=1:trialNum

%  [TrainingTime(ii), TestingTime(ii), trainAccuary(ii), testAccuary(ii)] = elm_kernel(TrainingLData, TestingData, 1, 10^3, 'RBF_kernel',1);
 model = elm_kernel_train(TrainingLData,'RBF_kernel');
 Output1 = elm_kernel_test(TrainingLData, model);
 Output2 = elm_kernel_test(TestingData, model);
 t_sim1=Output1.Result(1:end,1:end)';
 t_sim2=Output2.Result(1:end,1:end)';
%  TrainingTime(ii)=model.TrainingTime;
%  TestingTime(ii)=Output2.TestingTime;
%  trainAccuary(ii)=Output1.TestingAccuracy;
%  testAccuary(ii)=Output2.TestingAccuracy;
%end

% aveTrainingTime=mean(TrainingTime)
% aveTestingTime=mean(TestingTime)
% avetrainAccuary=sum(trainAccuary)/trialNum
% vartrainAccuary=var(trainAccuary)
% avetestAccuary=sum(testAccuary)/trialNum
% vartestAccuary=var(testAccuary)
%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
%%
for ii=1:N
    count=0;
    for jj=1:n
        if abs(T_test(jj,ii)-T_sim2(jj,ii))<0.05*abs(T_test(jj,ii))
            count=count+1;
        end
    end
    accuary(ii,:)=count/n;
end
accuary_total = mean(accuary);
%%  绘图
% figure
% plot(1: 97, T_train(:,5), 'r-*', 1: 97, T_sim1(:,5), 'b-o', 'LineWidth', 1)
% xlabel('预测样本')
% ylabel('预测结果')
% xlim([1, 97])
% grid

figure
plot(1: 97, T_test(:,1), 'r-*', 1: 97, T_sim2(:,1), 'b-o', 'LineWidth', 1)
xlabel('预测样本')
ylabel('预测结果')
xlim([1, 97])
grid
% 
% figure
% plot(1: 97, T_test(:,2), 'r-*', 1: 97, T_sim2(:,2), 'b-o', 'LineWidth', 1)
% xlabel('预测样本')
% ylabel('预测结果')
% xlim([1, 97])
% grid
% figure
% plot(1: 97, T_test(:,3), 'r-*', 1: 97, T_sim2(:,3), 'b-o', 'LineWidth', 1)
% xlabel('预测样本')
% ylabel('预测结果')
% xlim([1, 97])
% grid
% figure
% plot(1: 97, T_test(:,4), 'r-*', 1: 97, T_sim2(:,4), 'b-o', 'LineWidth', 1)
% xlabel('预测样本')
% ylabel('预测结果')
% xlim([1, 97])
% grid
% figure
% plot(1: 97, T_test(:,6), 'r-*', 1: 97, T_sim2(:,6), 'b-o', 'LineWidth', 1)
% xlabel('预测样本')
% ylabel('预测结果')
% xlim([1, 97])
% grid