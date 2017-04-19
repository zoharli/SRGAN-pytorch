import argparse
import os
import shutil
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from models import *
from dataset import *
import numpy as np
from setproctitle import *

parser = argparse.ArgumentParser(description='Pretraining')
parser.add_argument('-j', '--workers', default=4, type=int,
        help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=40, type=int,
        help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int,
        help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
        help='mini-batch size (default: 16)')
parser.add_argument('--crop-size','-c',default=256,type=int,
        help='crop size of the hr image')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
        help='initial learning rate')
parser.add_argument('--momentum','-m',default=0.9,type=float,
        help='momentum if using sgd optimization')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
        help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
        help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str,
        help='path to latest checkpoint (default: none)')
parser.add_argument('--logdir','-s',default='save',type=str,
        help='path to save checkpoint')
parser.add_argument('--optim','-o',default='RMSprop',
        help='the optimization method to be employed')
parser.add_argument('--traindir',default='r375-400.bin',
        help=' the global name of training set dir')
parser.add_argument('--valdir',default='b100',type=str,
        help='the global name of validation set dir')

best_psnr = -100
args = parser.parse_args()
args.__dict__['upscale_factor']=4
#args.traindir=globals()[args.traindir]
args.valdir=globals()[args.valdir]
args.__dict__['model_base_name']='srgan_%s'%args.optim
args.__dict__['model_name']=args.model_base_name+'.pth'
args.__dict__['snapshot']='snapshot_'+args.model_base_name

setproctitle('train_'+args.model_base_name)
if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)
if not os.path.exists(args.snapshot):
    os.makedirs(args.snapshot)

cudnn.benchmark = True

train_loader = torch.utils.data.DataLoader(
    get_train_set(args),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)

gen=GenNet().cuda()
disc=DisNet().cuda()
vgg=vgg19_54().cuda()

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_psnr = checkpoint['best_psnr']
        gen.load_state_dict(checkpoint['gen_state_dict'])
        disc.load_state_dict(checkpoint['disc_state_dict'])
        print("=> loaded checkpoint (epoch {})"
                .format( checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

criterion = nn.MSELoss().cuda()

gen_optimizer = torch.optim.RMSprop(gen.parameters(), args.lr)
disc_optimizer = torch.optim.RMSprop(disc.parameters(),args.lr)

def rgb2y_matlab(img):   #convert a PIL rgb image to y-channel image in `matlab` way
    img=(img/255.0)[4:-4,4:-4,:]
    img=65.481*img[:,:,0]+128.553*img[:,:,1]+24.966*img[:,:,2]+16.0
    return img

def validate(model,criterion,valdir,epoch,factor,optimizer):
    cnt=0
    sum=0
    ysum=0
    for x in [os.path.join(valdir,y) for y in os.listdir(valdir)]:
        im=Image.open(x)
        im=im.resize((im.size[0]-im.size[0]%factor,im.size[1]-im.size[1]%factor),Image.BICUBIC)
        target=Variable(torch.stack([transforms.ToTensor()(im)],0),volatile=True).cuda()
        input=torch.stack([transforms.ToTensor()(im.resize((im.size[0]//args.upscale_factor,im.size[1]//args.upscale_factor),Image.BICUBIC))],0)
        input_var=Variable(input,volatile=True).cuda()
        output=model(input_var)
        loss=criterion(target,output).cpu().data[0]
        img=torch.squeeze(output.data.cpu())
        rgb=transforms.ToPILImage()(img)
        rgb.save(os.path.join(args.snapshot,os.path.basename(x)))
        yorigin=rgb2y_matlab(np.asarray(im))
        youtput=rgb2y_matlab(np.asarray(rgb))
        ymse=np.mean((yorigin-youtput)**2)
        ypsnr=10*(np.log10(255.0**2/ymse))
        psnr=10*np.log10(1.0/loss)
        sum+=psnr
        ysum+=ypsnr
        cnt+=1
    psnr=float(sum)/cnt
    ypsnr=float(ysum)/cnt
    lr=optimizer.param_groups[0]['lr']
    s=time.strftime('%dth-%H:%M:%S',time.localtime(time.time()))+'===>epoch%d=>lr=%.6f=>psnr=%.3f=>ypsnr=%.3f'%(epoch,lr,psnr,ypsnr)
    return s,psnr
    
def save_checkpoint(state, is_best,logdir):
    filename=os.path.join(logdir,args.model_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(logdir,'best_'+args.model_name))

def train(train_loader, gen,disc,gen_optimizer,disc_optimizer,ContentLoss,epoch,args):
    for i, (input, target) in enumerate(train_loader):
        input_var = Variable(input.cuda())
        target_var = Variable(target.cuda())
        

        real_loss=((disc(target_var)-1)**2).mean()
        disc_optimizer.zero_grad()
        real_loss.backward()
        disc_optimizer.step()

        fake_loss=torch.mean(disc(gen(input_var))**2)
        disc_optimizer.zero_grad()
        fake_loss.backward()
        disc_optimizer.step()
        disc_loss=(fake_loss+real_loss)/2

        G_z=gen(input_var)
        fake_feature=vgg(G_z)
        real_feature=vgg(target_var).detach()
        content_loss=criterion(fake_feature,real_feature)
        adv_loss=torch.mean((disc(G_z)-1)**2)
        gen_loss=adv_loss+content_loss
        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()
        
        if i % args.print_freq == 0:
            s,vpsnr=validate(gen,criterion,args.valdir,epoch,args.upscale_factor,gen_optimizer)
            s+='=>gen_loss%.3f[C%.3f/A%.3f]=>disc_loss%.3f[R%.3f/F%.3f]'%(gen_loss.data[0],content_loss.data[0],adv_loss.data[0],disc_loss.data[0],real_loss.data[0],fake_loss.data[0])
            print(s)
            f=open('info.'+args.model_base_name,'a')
            f.write(s+'\n')
            f.close()
            global best_psnr
            is_best = vpsnr > best_psnr
            best_psnr = max(vpsnr, best_psnr)
            save_checkpoint({
                'epoch': epoch + 1,
                'gen_state_dict': gen.state_dict(),
                'disc_state_dict':disc.state_dict(),
                'best_psnr': best_psnr,
            }, is_best,args.logdir)

for epoch in range(args.start_epoch, args.epochs):
    train(train_loader,gen,disc,gen_optimizer,disc_optimizer,criterion,epoch,args)
