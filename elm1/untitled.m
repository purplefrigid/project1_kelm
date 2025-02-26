if size(p_train, 2) ~= size(t_train, 2)
    error('ELM:Arguments', 'The columns of P and T must be same.');
end

%%  转入分类模式
if TYPE  == 1
    t_train  = ind2vec(t_train);
end

%%  初始化权重
R = size(p_train, 1);
Q = size(t_train, 2);
IW = rand(N, R) * 2 - 1;
B  = rand(N, 1);
BiasMatrix = repmat(B, 1, Q);

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
