function [IW, B, LW, TF, TYPE] = elmtrain(p_train, t_train, N, TF, TYPE)
% P   - Input Matrix of Training Set  (R * Q)
% T   - Output Matrix of Training Set (S * Q)
% N   - Number of Hidden Neurons (default = Q)
% TF  - Transfer Function:
%       'sig' for Sigmoidal function (default)
%       'hardlim' for Hardlim function
% TYPE - Regression (0, default) or Classification (1)
% Output
% IW  - Input Weight Matrix (N * R)
% B   - Bias Matrix  (N * 1)
% LW  - Layer Weight Matrix (N * S)

if size(p_train, 2) ~= size(t_train, 2)
    error('ELM:Arguments', 'The columns of P and T must be same.');
end
%%  节点个数
R  = size(p_train, 1);  % 输入层节点数
Q = size(t_train, 1);  % 输出层节点数
NumberofTrainingData=size(p_train,2);
%%  参数初始化
c1      = 4.494;       % 学习因子
c2      = 4.494;       % 学习因子
maxgen  =   1500;        % 种群更新次数  
sizepop =    200;        % 种群规模
Vmax    =  0.001;        % 最大速度
Vmin    = -0.001;        % 最小速度
popmax  =  2.0;        % 最大边界
popmin  = -2.0;        % 最小边界
%%  节点总数
numsum = N*R+N;

for i = 1 : sizepop
    pop(i, :) = rands(1, numsum);  % 初始化种群
    V(i, :) = rands(1, numsum);    % 初始化速度
    fitness(i) = fun(pop(i, :), p_train, t_train, N, TF);
end
%%  个体极值和群体极值
[fitnesszbest, bestindex] = min(fitness);
zbest = pop(bestindex, :);     % 全局最佳
gbest = pop;                   % 个体最佳
fitnessgbest = fitness;        % 个体最佳适应度值
BestFit = fitnesszbest;        % 全局最佳适应度值
%%  迭代寻优
for i = 1 : maxgen
    for j = 1 : sizepop
        
        % 速度更新
        V(j, :) = V(j, :) + c1 * rand * (gbest(j, :) - pop(j, :)) + c2 * rand * (zbest - pop(j, :));
        V(j, (V(j, :) > Vmax)) = Vmax;
        V(j, (V(j, :) < Vmin)) = Vmin;
        
        % 种群更新
        pop(j, :) = pop(j, :) + 0.2 * V(j, :);
        pop(j, (pop(j, :) > popmax)) = popmax;
        pop(j, (pop(j, :) < popmin)) = popmin;
        
        % 自适应变异
        pos = unidrnd(numsum);
        if rand > 0.85
            pop(j, pos) = rands(1, 1);
        end
        
        % 适应度值
        fitness(j) = fun(pop(j, :), p_train, t_train, N, TF);
        
    end
    
    for j = 1 : sizepop

        % 个体最优更新
        if fitness(j) < fitnessgbest(j)
            gbest(j, :) = pop(j, :);
            fitnessgbest(j) = fitness(j);
        end

        % 群体最优更新 
        if fitness(j) < fitnesszbest
            zbest = pop(j, :);
            fitnesszbest = fitness(j);
        end

    end

    BestFit = [BestFit, fitnesszbest];    
end
%%  提取权值和阈值
zbest_sub=pop(1:N*R);
IW = reshape(zbest_sub, N, R);
B  = zbest(N*R+1:end)';
%BiasMatrix = repmat(B, 1, Q);
ind=ones(1,NumberofTrainingData);     % 训练集中分类对象的个数
BiasMatrix=B(:,ind);% 扩展BiasMatrix矩阵大小与H匹配 
%%  计算输出
tempH = IW * p_train + BiasMatrix;

%%  选择激活函数
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'hardlim'
        H = hardlim(tempH);
end

%%  伪逆计算权重
LW = pinv(H') * t_train';