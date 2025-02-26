function model = elm_kernel_train(TrainingData,C,Kernel_type, Kernel_para)

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
dataLabel=train_data(:,1);
data=train_data(:,2:end);
clear train_data;                                   %   Release raw training data array

NumberofTrainingData=size(data,1);
    %%%%%%%%%%%% Preprocessing the data of classification
    label=unique(dataLabel);
    number_class=numel(label);
    NumberofOutputNeurons=number_class;
    model.label=label;
    model.X=data;
    %%%%%%%%%% Processing the targets of training
    temp=zeros(NumberofTrainingData,NumberofOutputNeurons );
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(j) == dataLabel(i)
                break; 
            end
        end
        temp(i,j)=1;
    end
    dataLabelMP=temp*2-1;%转化成多节点形式

    %%%%%%%%%% Processing the targets of testing
 


%%%%%%%%%%% Training Phase %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
n = size(dataLabel,1);
Omega_train = kernel_matrix(data,Kernel_type, Kernel_para);
model.beta=((Omega_train+speye(n)/C)\(dataLabelMP)); 
model.TrainingTime=toc;
model.Kernel_type=Kernel_type;
model.Kernel_para=Kernel_para;
end
    
    
