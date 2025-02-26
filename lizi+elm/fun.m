function error = fun(pop, p_train, t_train, N, TF)

%%  �ڵ����
R  = size(p_train, 1);  % �����ڵ���
Q = size(t_train, 1);  % �����ڵ���
NumberofTrainingData=size(p_train,2);
%%  ��ȡȨֵ����ֵ
pop_sub=pop(1:N*R);
IW = reshape(pop_sub, N, R);
B  = pop(N*R+1:end)';
%BiasMatrix = repmat(B, 1, Q);
ind=ones(1,NumberofTrainingData);     % ѵ�����з������ĸ���
BiasMatrix=B(:,ind);% ��չBiasMatrix�����С��Hƥ�� 
%%  �������
tempH = IW * p_train + BiasMatrix;

%%  ѡ�񼤻��
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'hardlim'
        H = hardlim(tempH);
end

%%  α�����Ȩ��
LW = pinv(H') * t_train';

%%  �������
t_sim = elmpredict(p_train, IW, B, LW, TF, 0);

%%  ��Ӧ��ֵ
error = sum(sqrt(sum((t_sim - t_train) .^ 2) ./ length(t_sim)));