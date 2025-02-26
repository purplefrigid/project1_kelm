function error = error_test(pop, train_data,Kernel_type)
%%  提取权值和阈值
C=pop(1)*(10^8);
Kernel_para=pop(2)/10;
dataLabel=train_data(:,1:97);
data=train_data(:,98:end);
n = size(dataLabel,1);

model.label=dataLabel;
model.X=data;

Omega_train = kernel_matrix(data,Kernel_type, Kernel_para);
model.beta=((Omega_train+speye(n)/C)\(dataLabel));  
%model.TrainingTime=toc;
model.Kernel_type=Kernel_type;
model.Kernel_para=Kernel_para;
                        
%%  仿真测试
Output = elm_kernel_test(train_data, model);
a=(Output.Result(1:end,1:end) - dataLabel);
b=a .^2;
c=length(dataLabel);
d=sum(b);
e=d ./c;
f=sqrt(e);
g=sum(f);
%%  适应度值
error = sum(sqrt(sum((Output.Result(1:end,1:end) - dataLabel) .^ 2) ./ length(dataLabel)));