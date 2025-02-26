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
%%  �ڵ����
R  = size(p_train, 1);  % �����ڵ���
Q = size(t_train, 1);  % �����ڵ���
NumberofTrainingData=size(p_train,2);
%%  ������ʼ��
c1      = 4.494;       % ѧϰ����
c2      = 4.494;       % ѧϰ����
maxgen  =   1500;        % ��Ⱥ���´���  
sizepop =    200;        % ��Ⱥ��ģ
Vmax    =  0.001;        % ����ٶ�
Vmin    = -0.001;        % ��С�ٶ�
popmax  =  2.0;        % ���߽�
popmin  = -2.0;        % ��С�߽�
%%  �ڵ�����
numsum = N*R+N;

for i = 1 : sizepop
    pop(i, :) = rands(1, numsum);  % ��ʼ����Ⱥ
    V(i, :) = rands(1, numsum);    % ��ʼ���ٶ�
    fitness(i) = fun(pop(i, :), p_train, t_train, N, TF);
end
%%  ���弫ֵ��Ⱥ�弫ֵ
[fitnesszbest, bestindex] = min(fitness);
zbest = pop(bestindex, :);     % ȫ�����
gbest = pop;                   % �������
fitnessgbest = fitness;        % ���������Ӧ��ֵ
BestFit = fitnesszbest;        % ȫ�������Ӧ��ֵ
%%  ����Ѱ��
for i = 1 : maxgen
    for j = 1 : sizepop
        
        % �ٶȸ���
        V(j, :) = V(j, :) + c1 * rand * (gbest(j, :) - pop(j, :)) + c2 * rand * (zbest - pop(j, :));
        V(j, (V(j, :) > Vmax)) = Vmax;
        V(j, (V(j, :) < Vmin)) = Vmin;
        
        % ��Ⱥ����
        pop(j, :) = pop(j, :) + 0.2 * V(j, :);
        pop(j, (pop(j, :) > popmax)) = popmax;
        pop(j, (pop(j, :) < popmin)) = popmin;
        
        % ����Ӧ����
        pos = unidrnd(numsum);
        if rand > 0.85
            pop(j, pos) = rands(1, 1);
        end
        
        % ��Ӧ��ֵ
        fitness(j) = fun(pop(j, :), p_train, t_train, N, TF);
        
    end
    
    for j = 1 : sizepop

        % �������Ÿ���
        if fitness(j) < fitnessgbest(j)
            gbest(j, :) = pop(j, :);
            fitnessgbest(j) = fitness(j);
        end

        % Ⱥ�����Ÿ��� 
        if fitness(j) < fitnesszbest
            zbest = pop(j, :);
            fitnesszbest = fitness(j);
        end

    end

    BestFit = [BestFit, fitnesszbest];    
end
%%  ��ȡȨֵ����ֵ
zbest_sub=pop(1:N*R);
IW = reshape(zbest_sub, N, R);
B  = zbest(N*R+1:end)';
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