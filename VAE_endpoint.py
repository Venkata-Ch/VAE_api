import logging

import torch
import torchvision
from torch import nn, optim
from torchvision.tv_tensors import Image

from .VAE_model import VAE_arch
from torchvision import transforms,datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os

from fastapi import FastAPI,UploadFile,File
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from PIL import Image


from configparser import ConfigParser
con = ConfigParser()
con.read("./.config")
path = con.get('model_path','path')

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inp_ = con.get('model_config','INP_DIM')
hidden = con.get('model_config','DIMH')
dim = con.get('model_config','DIM_')
epochs = con.get('model_config','EPOCH_NUM')
batch_size = con.get('model_config','BATCH_')
learning_rate = con.get('model_config','LR_RATE')



#Loading the model
vae_model = VAE_arch(inp_,hidden,dim).to(dev)
vae_model.load_state_dict(torch.load('model/Variaional_auto.pt',map_location=dev,weights_only=True))
vae_model.eval()

#Loading the dataset
dataset = datasets.MNIST(root="data_set/",train=True,transform=transforms.ToTensor(),download=False)

app = FastAPI(title="VAE Model")
app.mount("/static",StaticFiles(directory='static'),name="static")


class VAE_Endpoint:

    def __init(self):
        pass


    @app.post("/img_upload")
    async def img(file: UploadFile= File(...)):
        try:
            picture = Image.open(file.file).convert("L")
            picture = picture.resize((28, 28))
            picture = transforms.ToTensor()(picture).view(1,784).to(dev)

            with torch.no_grad():
                phi,epsilon = vae_model.encode(picture)

            images_created = []
            for example in range(5):  # Generate 5 examples
                delta = torch.randn_like(epsilon)
                z = phi + epsilon *  delta
                out = vae_model.decode(z)
                out_img_path = f"static/generated_example_{example}.png"
                save_image(out.view(-1, 1, 28, 28), out_img_path)  # Save generated image

                images_created.append(out_img_path)  # Store path for response

            return {"generated_images": images_created}
            return FileResponse("your_image.jpeg")
        except Exception as error:
            logging.error(str(error))



def main():
    API = VAE_Endpoint()
    API.img()


if __name__=="__main__":
    main()











