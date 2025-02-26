function Output = elm_kernel_test(TestingData, model)

% Usage: Output = elm_kernel_test(TestingData, model)

%
% Input:
% TestingData_File             - m*n training data with m instances and
%                                n-1 features and the first column represents the labels
% model                         -model trained by elm_kernel_train
% Output: 
%   Output                     -structure data
%                                  Output1.PredictLabel           PredictLabel                         
%                                  Output1.TestingAccuracy          TestingAccuracy
%                                  Output1.TestingTime              TestingTime
%

%Author: Yin Haibo
%Date :2015 10 25
%Reference on:
    %%%%    Authors:    MR HONG-MING ZHOU AND DR GUANG-BIN HUANG
    %%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
    %%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
    %%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
    %%%%    DATE:       MARCH 2012





%%%%%%%%%%% Load testing dataset
test_data=TestingData;
TestLabel=test_data(:,1);
TestData=test_data(:,2:end);



   

clear test_data;                                    %   Release raw testing data array
                                   

tic;
Kernel_type=model.Kernel_type;
Kernel_para=model.Kernel_para;
Omega_test = kernel_matrix(TestData,Kernel_type, Kernel_para,model.X);
TY=Omega_test * model.beta;
model.TrainingTime=toc;
MissClassificationRate_Testing=0;

[~,PredictLabelInx]=max(TY,[],2);
Output.PredictLabel=model.label(PredictLabelInx);


for i = 1 : size(TY, 1)
        if Output.PredictLabel(i)~=TestLabel(i)
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
end
 Output.TestingAccuracy=1-MissClassificationRate_Testing/size(TY, 1);
 Output.TestingTime=toc;   
end

    
