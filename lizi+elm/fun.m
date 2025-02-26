function error = fun(pop, p_train, t_train, N, TF)

%%  节点个数
R  = size(p_train, 1);  % 输入层节点数
Q = size(t_train, 1);  % 输出层节点数
NumberofTrainingData=size(p_train,2);
%%  提取权值和阈值
pop_sub=pop(1:N*R);
IW = reshape(pop_sub, N, R);
B  = pop(N*R+1:end)';
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

%%  仿真测试
t_sim = elmpredict(p_train, IW, B, LW, TF, 0);

%%  适应度值
error = sum(sqrt(sum((t_sim - t_train) .^ 2) ./ length(t_sim)));