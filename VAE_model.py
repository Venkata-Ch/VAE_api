import torch
import torchvision.datasets as datasets
import pickle
from tqdm import tqdm
from torch import nn,optim
from .VAE_ import VAE_arch
from torchvision import transforms
from torch.utils.data import DataLoader

#Model Config
EXCECUTER = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INP_DIM = 784
DIMH = 200
DIM_ = 20
EPOCH_NUM = 10
BATCH_ = 32
LR_RATE = 3e-4

#Loading the dataset
data = datasets.MNIST(root="data_set/",train=True, transform=transforms.ToTensor(),download=True)
load_train_set = DataLoader(dataset=data,batch_size=BATCH_,shuffle=True)
model = VAE_arch(INP_DIM,DIMH,DIM_).to(EXCECUTER)
optimizer = optim.Adam(model.parameters(),lr=LR_RATE)
loss_function = nn.BCELoss(reduction="sum")

for epoch in range(EPOCH_NUM):
    model.train()
    loop = tqdm(enumerate(load_train_set))
    for i, (x, _) in loop:
        x = x.to(EXCECUTER).view(x.shape[0],INP_DIM)
        # x_param = torch.sigmoid(x_param)
        x_param,phi,epsilon = model(x)
        # print(torch.min(x_param), torch.max(x_param))
        #Loss Computation
        x_param = torch.sigmoid(x_param)
        reconstruct_loss = loss_function(x_param,x)
        kl_divergence = -torch.sum(1 + torch.log(epsilon.pow(2)) - phi.pow(2) - epsilon.pow(2))

        #Backpropogation
        loss = reconstruct_loss + kl_divergence
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

torch.save(model.state_dict(),'model/Variaional_auto.pt')





