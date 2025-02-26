function model = elm_kernel_train(TrainingData,Kernel_type)

% Usage: model = elm_kernel_train(TrainingData,C,Kernel_type, Kernel_para)

%
% Input:
% TrainingData           - m*n training data with m instances and
%                          n-1 features and the first column represents the labels
% 

% C  - Regularization coefficient C(usually very small)
% Kernel_type                 - Type of Kernels:
%                                   'RBF_kernel' for RBF Kernel
%                                   'lin_kernel' for Linear Kernel
%                                   'poly_kernel' for Polynomial Kernel
%                                   'wav_kernel' for Wavelet Kernel
%Kernel_para                  - A number or vector of Kernel Parameters. eg. 1, [0.1,10]...
% Output: 
% model                       -stucture data containing model information
%
% MULTI-CLASSE CLASSIFICATION: NUMBER OF OUTPUT NEURONS WILL BE AUTOMATICALLY SET EQUAL TO NUMBER OF CLASSES
% FOR EXAMPLE, if there are 7 classes in all, there will have 7 output
% neurons; neuron 5 has the highest output means input belongs to 5-th class

%Authors: Yin Haibo
% Date:2015 10 25
% Reference on
    %%%%    Authors:    MR HONG-MING ZHOU AND DR GUANG-BIN HUANG
    %%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
    %%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
    %%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
    %%%%    DATE:       MARCH 2012


%%%%%%%%%%% Load training dataset
train_data=TrainingData;
dataLabel=train_data(:,1:97);
n = size(dataLabel,1);
%%%%%%%%%%% Training Phase %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tic;

%%  ������ʼ��
c1      = 0.03;       % ѧϰ����
c2      = 0.03;       % ѧϰ����
maxgen  =   500;        % ��Ⱥ���´���  
sizepop =    50;        % ��Ⱥ��ģ
Vmax    =  0.4;        % ����ٶ�
Vmin    = -0.4;        % ��С�ٶ�
popmax  =  10.0;        % ���߽�
popmin  = -10.0;        % ��С�߽�
%%  �ڵ�����
numsum = 2;

for i = 1 : sizepop
    pop(i, :) = randi([1, 10], 1, 2);  % ��ʼ����Ⱥ
    %pop(i, :) = [1,1];
    V(i, :) = 0.1*rands(1, numsum);    % ��ʼ���ٶ�
    fitness(i) = error_test(pop(i, :), train_data,Kernel_type);
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
        fitness(j) = error_test(pop(j, :), train_data,Kernel_type);
        
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
C = zbest(1)*(10^8);
Kernel_para  = zbest(2)/10;
% C = 10^9;
% Kernel_para  = 1;
%%  �������
dataLabel=train_data(:,1:97);
data=train_data(:,98:end);
model.label=dataLabel;
model.X=data;

Omega_train = kernel_matrix(data,Kernel_type, Kernel_para);
model.beta=((Omega_train+speye(n)/C)\(dataLabel)); 
%model.beta=(dataLabel\(Omega_train+speye(n)/C))'; 
%model.TrainingTime=toc;
model.Kernel_type=Kernel_type;
model.Kernel_para=Kernel_para;

end
    
    
