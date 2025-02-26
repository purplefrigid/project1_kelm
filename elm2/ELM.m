%function [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = ELM(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
%% ELM 算法程序
% 调用方式: [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
%
% 输入:
% TrainingData_File     - 训练数据集文件名
% TestingData_File      - 测试训练集文件名
% Elm_Type              - 任务类型：0 时为回归任务，1 时为分类任务
% NumberofHiddenNeurons - ELM的隐层神经元数目
% ActivationFunction    - 激活函数类型:
%                           'sig' ， Sigmoidal 函数
%                           'sin' ， Sine 函数
%                           'hardlim' ， Hardlim 函数
%                           'tribas' ， Triangular basis 函数
%                           'radbas' ， Radial basis 函数
% 输出: 
% TrainingTime          - ELM 训练花费的时间（秒）
% TestingTime           - 测试数据花费的时间（秒）
% TrainingAccuracy      - 训练的准确率（回归任务时为RMSE，分类任务时为分类正确率）                       
% TestingAccuracy       - 测试的准确率（回归任务时为RMSE，分类任务时为分类正确率）
%
% 调用示例（回归）: [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = ELM('sinc_train', 'sinc_test', 0, 20, 'sig')
% 调用示例（分类）: [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = ELM('diabetes_train', 'diabetes_test', 1, 20, 'sig')


%% 数据预处理

% 定义任务类型
REGRESSION=0;
CLASSIFIER=1;

% 载入训练数据集
train_data=xlsread(TrainingData_File);
T=train_data(:,9:size(train_data,2))';                   % 第一列：分类或回归的期望输出
P=train_data(:,1:8)';% 第二列到最后一列：不同数据的属性
clear train_data;                     % 清除中间变量

% 载入测试数据集
test_data=xlsread(TestingData_File);
TV.T=test_data(:,9:size(test_data,2))';                 % 第一列：分类或回归的期望输出
TV.P=test_data(:,1:8)';% 第二列到最后一列：不同数据的属性
clear test_data;                       % 清除中间变量

% 获取训练、测试数据情况
NumberofTrainingData=size(P,2);        % 训练数据中分类对象个数
NumberofTestingData=size(TV.P,2);      % 测试数据中分类对象个数
NumberofInputNeurons=size(P,1);        % 神经网络输入个数，训练数据的属性个数
n=size(T,1);
%% 分类任务时的数据编码
% if Elm_Type~=REGRESSION
%     % 分类任务数据预处理
%     sorted_target=sort(cat(2,T,TV.T),2);% 将训练数据和测试数据的期望输出合并成一行，然后按从小到大排序
%     label=zeros(1,1);                   %  Find and save in 'label' class label from training and testing data sets
%     label(1,1)=sorted_target(1,1);      % 存入第一个标签
%     j=1;
%     % 遍历所有数据集标签（期望输出）得到数据集的分类数目
%     for i = 2:(NumberofTrainingData+NumberofTestingData)
%         if sorted_target(1,i) ~= label(1,j)
%             j=j+1;
%             label(1,j) = sorted_target(1,i);
%         end
%     end
%     number_class=j;                    % 统计数据集（训练数据和测试数据）一共有几类
%     NumberofOutputNeurons=number_class;% 一共有几类，神经网络就有几个输出
%        
%     % 预定义期望输出矩阵
%     temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
%     % 遍历所有训练数据的标记，扩充为num_class*NumberofTraingData的矩阵 
%     for i = 1:NumberofTrainingData
%         for j = 1:number_class
%             if label(1,j) == T(1,i)
%                 break; 
%             end
%         end
%         temp_T(j,i)=1;                %一个矩阵，行是分类，列是对象，如果该对象在此类就置1
%     end
%     T=temp_T*2-1;                     % T为处理的期望输出矩阵，每个对象（列）所在的真实类（行）位置为1，其余为-1
% 
%     % 遍历所有测试数据的标记，扩充为num_class*NumberofTestingData的矩阵 
%     temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
%     for i = 1:NumberofTestingData
%         for j = 1:number_class
%             if label(1,j) == TV.T(1,i)
%                 break; 
%             end
%         end
%         temp_TV_T(j,i)=1;            % 期望输出表示矩阵，行是分类，列是对象，如果该对象在此类就置1
%     end
%     TV.T=temp_TV_T*2-1;              % T为处理的期望输出矩阵，每个对象（列）所在的真实类（行）位置为1，其余为-1
% 
% 
% end  % Elm_Type

%% 计算隐藏层的输出H
start_time_train=cputime;           % 训练开始计时
for NumberofHiddenNeurons=5:80
% 随机产生输入权值InputWeight (w_i)和隐层偏差biases BiasofHiddenNeurons (b_i)
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1; % 输入节点的权重在[-1,1]之间
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);                % 连接偏重在[0,1]之间
tempH=InputWeight*P; % 不同对象的属性*权重
%clear P; % 释放空间 
ind=ones(1,NumberofTrainingData);     % 训练集中分类对象的个数
BiasMatrix=BiasofHiddenNeurons(:,ind);% 扩展BiasMatrix矩阵大小与H匹配 
tempH=tempH+BiasMatrix;               % 加上偏差的最终隐层输入

%计算隐藏层输出矩阵
switch lower(ActivationFunction) % 选择激活函数，lower是将字母统一为小写
    case {'sig','sigmoid'}
        H = 1 ./ (1 + exp(-tempH));% Sigmoid 函数
    case {'sin','sine'}
        H = sin(tempH);            % Sine 函数
    case {'hardlim'}
        H = double(hardlim(tempH));% Hardlim 函数
    case {'tribas'}
        H = tribas(tempH);         % Triangular basis 函数
    case {'radbas'}
        H = radbas(tempH);         % Radial basis 函数
    % 可在此添加更多激活函数                
end
clear tempH;% 释放不在需要的变量

%% 计算输出权重 OutputWeight (beta_i)
OutputWeight=pinv(H') * T';   % 无正则化因子的应用，参考2006年 Neurocomputing 期刊上的论文
% OutputWeight=inv(eye(size(H,1))/C+H * H') * H * T';   % faster method 1 ，参考 2012 IEEE TSMC-B 论文
% OutputWeight=(eye(size(H,1))/C+H * H') \ H * T';      % faster method 2 ，refer to 2012 IEEE TSMC-B 论文

end_time_train=cputime;
TrainingTime=end_time_train-start_time_train; % 计算训练ELM时CPU花费的时间

% 计算输出
Y=(H' * OutputWeight)';                       % Y为训练数据输出（列向量） 
if Elm_Type == REGRESSION 
    TrainingAccuracy=sqrt(mse(T - Y));        % 回归问题计算均方误差根
end
clear H;

%% 计算测试数据的输出（预测标签）
start_time_test=cputime;    % 测试计时
tempH_test=InputWeight*TV.P;% 测试的输入
clear TV.P;  

ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind); % 扩展BiasMatrix矩阵大小与H匹配 
tempH_test=tempH_test + BiasMatrix;% 加上偏差的最终隐层输入
switch lower(ActivationFunction)
    case {'sig','sigmoid'}% Sigmoid 函数   
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}   % Sine 函数 
        H_test = sin(tempH_test);        
    case {'hardlim'}      % Hardlim 函数
        H_test = hardlim(tempH_test);        
    case {'tribas'}       % Triangular basis 函数
         H_test = tribas(tempH_test);        
    case {'radbas'}       % Radial basis 函数
         H_test = radbas(tempH_test);        
    % 可在此添加更多激活函数             
end
TY=(H_test' * OutputWeight)';                       %   TY: 测试数据的输出

end_time_test=cputime;
TestingTime=end_time_test-start_time_test;          % 计算ELM测试集时CPU花费的时间

 %% 计算准确率
% if Elm_Type == REGRESSION
%     TestingAccuracy=sqrt(mse(TV.T - TY));           % 回归问题计算均方误差根
% end
% 
% % 如果是分类问题计算分类的准确率
% if Elm_Type == CLASSIFIER 
%     MissClassificationRate_Training=0;
%     MissClassificationRate_Testing=0;
%     % 计算训练集上的分类准确率
%     for i = 1 : size(T, 2) 
%         [x, label_index_expected]=max(T(:,i));
%         [x, label_index_actual]=max(Y(:,i));
%         if label_index_actual~=label_index_expected
%             MissClassificationRate_Training=MissClassificationRate_Training+1;
%         end
%     end
%     % 计算测试集上的分类准确率
%     TrainingAccuracy=1-MissClassificationRate_Training/size(T,2); % 训练集分类正确率
%     for i = 1 : size(TV.T, 2)
%         [x, label_index_expected]=max(TV.T(:,i));
%         [x, label_index_actual]=max(TY(:,i));
%         if label_index_actual~=label_index_expected
%             MissClassificationRate_Testing=MissClassificationRate_Testing+1;
%         end
%     end
%     TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2);  % 测试集分类正确率
% end
%%
for ii=1:NumberofTestingData
    count=0;
    for jj=1:n
        if abs(TV.T(jj,ii)-TY(jj,ii))<0.1*abs(TY(jj,ii))
            count=count+1;
        end
    end
    accuary(ii,:)=count/n;
end
accuary_total(NumberofHiddenNeurons,:) = mean(accuary);
end
%%  绘图
figure
plot(1: 351, Y(:,1), 'b-o', 1: 351, T(:,1), 'r-*', 'LineWidth', 1)
xlabel('预测样本')
ylabel('预测结果')
xlim([1, 351])
grid

figure
plot(1: 351, TY(:,6), 'b-o', 1: 351, TV.T(:,6), 'r-*', 'LineWidth', 1)
xlabel('预测样本')
ylabel('预测结果')
xlim([1, 351])
grid

figure
plot(1: 351, TY(:,2), 'b-o', 1: 351, TV.T(:,2), 'r-*', 'LineWidth', 1)
xlabel('预测样本')
ylabel('预测结果')
xlim([1, 351])
grid
figure
plot(1: 351, TY(:,3), 'b-o', 1: 351, TV.T(:,3), 'r-*', 'LineWidth', 1)
xlabel('预测样本')
ylabel('预测结果')
xlim([1, 351])
grid
figure
plot(1: 351, TY(:,4), 'b-o', 1: 351, TV.T(:,4), 'r-*', 'LineWidth', 1)
xlabel('预测样本')
ylabel('预测结果')
xlim([1, 351])
grid
figure
plot(1: 351, TY(:,5), 'b-o', 1: 351, TV.T(:,5), 'r-*', 'LineWidth', 1)
xlabel('预测样本')
ylabel('预测结果')
xlim([1, 351])
grid