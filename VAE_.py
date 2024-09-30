import torch
import torch.nn.functional as util
from torch import nn
from fastapi import FastAPI
import pickle

app  = FastAPI()
#Standard
class VAE_arch(nn.Module):
    def __init__(self, inp_dim,dimh=200,dim_=30):
        super().__init__()

        #Encoder
        self.dimh_img = nn.Linear(int(inp_dim),int(dimh))
        self.hi_2mu = nn.Linear(int(dimh),int(dim_))
        self.h_de = nn.Linear(int(dimh),int(dim_))

        #Decode
        self.h3_im = nn.Linear(int(dim_),int(dimh))
        self.h_e = nn.Linear(int(dimh),int(inp_dim))

        #activation function
        self.relu = nn.ReLU()

    def encode(self, x):
        a = self.relu(self.dimh_img(x))
        phi,epsilon = self.hi_2mu(a), self.h_de(a)
        return phi, epsilon


    def decode(self,y):
        b = self.relu(self.h3_im(y))
        return torch.sigmoid(self.h_e(b))

    def forward(self,x):
        phi,epsilon = self.encode(x)
        kappa = torch.randn_like(epsilon)
        y_param = phi+kappa*epsilon
        x_param = self.decode(y_param)
        return x_param,phi,epsilon





if __name__=="__main__":
    x = torch.randn(4,28*28)

    v = VAE_arch(inp_dim=784)
    x_param,phi,epsilon = v(x)
    print(x_param.shape)
    print(phi.shape)
    print(epsilon.shape)
    # print(list(v(x).shape()))


