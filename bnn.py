import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import DataLoader, Dataset  
import pyro  
import pyro.distributions as dist  
from pyro.nn import PyroSample   
from pyro.infer import SVI, Trace_ELBO  
from pyro.optim import ClippedAdam   
from pyro.nn.module import  PyroParam
from F_Dataset_8paras import MH_Data
from F_model_8paras import F_Net_1D,F_Net_2D

BATCH_SIZE = 200  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
Project_folder = ".\\" 
mh_train = MH_Data(Project_folder + 'train.xlsx')
mh_val = MH_Data(Project_folder + 'test.xlsx')
train_loader = DataLoader(batch_size=BATCH_SIZE, dataset=mh_train, shuffle=True, drop_last=True,
                              num_workers=1, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(batch_size=20, dataset=mh_val, shuffle=True,
                            num_workers=1, pin_memory=True, persistent_workers=True)

# net =  F_Net_2D().to(device)  
# Project_folder = "D:\\keti\\zjh\\wl\\" 
model_pth = "F_train_5paras3_2000-[1112].pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
net = torch.load(Project_folder  + model_pth, map_location=DEVICE)
net1=F_Net_2D().to(device)

def model(x_data,y_data=None):
    priors = {}  
    for name, param in net.named_parameters():  
        # priors[name] = dist.Normal(torch.zeros_like(param), torch.ones_like(param)).to_event(param.dim())
        # a= torch.ones_like(param)
        tensor_with_0_1 = torch.full_like(param, 0.001)  
        priors[name] = dist.Normal(param,tensor_with_0_1).to_event(param.dim())
        # new_tensor = priors[name].sample() 
    # print(f"Parameter name1: {name}, Shape: {param}") 
    lifted_module = pyro.random_module("bnn", nn_module=net, prior=priors)
    # # print(priors)
    lifted_model = lifted_module() 
    
    # lifted_module=pyro.poutine.lift(net, prior=priors)
    # lifted_model=lifted_module(x_data)
    for name, param in lifted_model.named_parameters():  
        a=1
    # print(f"Parameter name1: {name}, Shape: {param}")
    x_data=x_data.to(device)
    preds = lifted_model(x_data)
    # preds1 = net(x_data)
    y_data = y_data.squeeze(1)  
    # print("Model output shape:", preds.shape)  # 打印模型输出的形状
    # print("y_data:", y_data.shape)  # 打印模型输出的形状
    # 定义观测分布  
    with pyro.plate("data", x_data.size(0)):  
        obs = pyro.sample("obs", dist.Normal(preds, 0.1).to_event(1), obs=y_data)  
    return obs  


# 定义导向函数（Guide）  
def guide(x, y):  
    # 定义变分分布的参数  
    priors = {}  
    for name, param in net.named_parameters():  
        # 使用 pyro.param 定义变分参数  
        # loc = pyro.param(f"{name}_loc", torch.randn_like(param).to(device)) 
        tensor_with_0_1 = torch.full_like(param, 0.001)
        loc = pyro.param(f"{name}_loc", param.to(device)) 
        # scale = pyro.param(f"{name}_scale", torch.ones_like(param).to(device), constraint=dist.constraints.positive).detach().requires_grad_() 
        scale = pyro.param(f"{name}_scale", tensor_with_0_1, constraint=dist.constraints.positive).detach().requires_grad_()  
        # print("loc:",loc.is_leaf)  # 检查 param 是否是叶子张量
        # print("scale:",scale.is_leaf)  # 检查 param 是否是叶子张量     
        priors[name] = dist.Normal(loc, scale).to_event(param.dim())  
    # 使用 pyro.random_module 加载模型  
    
    lifted_bnn = pyro.random_module("bnn", net, priors)  
    return lifted_bnn()
# 定义优化器和推断算法  
optimizer = ClippedAdam({"lr": 0.001})  
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())  

# 训练模型  
def train_model(model, guide, trainloader, valloader,svi, epochs=100):  
    for epoch in range(epochs):  
        train_total_loss = 0  
        val_total_loss = 0  
        for xt, yt in trainloader:  
            xt=xt.to(device) 
            yt=yt.to(device) 
            loss = svi.step(xt, yt)  
            train_total_loss += loss  
        print(f"Epoch {epoch+1}/{epochs}, train_Loss: {train_total_loss:.4f}")  
        for xv, yv in valloader:  
            xv=xv.to(device) 
            yv=yv.to(device) 
            loss = svi.evaluate_loss(xv, yv)  
            val_total_loss+= loss  
        print(f"Epoch {epoch+1}/{epochs}, val_Loss: {val_total_loss:.4f}")


# 开始训练  
if __name__ == '__main__':
    train_model(model, guide, train_loader,val_loader, svi)
    pyro.get_param_store().save("guide_params.pt")  