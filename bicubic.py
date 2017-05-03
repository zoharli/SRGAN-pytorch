import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import *
import math
from PIL import Image
import numpy as np 

dir=b100
upscale_factor=4
sum=0
ysum=0
cnt=0
if not os.path.exists('bicubic'):
    os.makedirs('bicubic')

for x in os.listdir(dir):
    img=Image.open(os.path.join(dir,x))
    bic=img.resize((img.size[0]//upscale_factor,img.size[1]//upscale_factor),Image.BICUBIC)
    bic=bic.resize((img.size[0],
        img.size[1]),Image.BICUBIC)
    yimg=np.asarray(img.convert('YCbCr'))[:,:,0]/255.0*(235-16)+16
    img=np.asarray(img)
    mse=np.mean((img-np.asarray(bic))**2)
    psnr=10*np.log10(255**2/mse)
    cnt+=1
    bic.save('bicubic/'+x)
    sum+=psnr
    print(psnr,sum/cnt)
