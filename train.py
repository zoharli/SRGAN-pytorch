import math
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
parser.add_argument('--start-epoch', default=0, type=int,
        help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
        help='mini-batch size (default: 16)')
parser.add_argument('--crop-size','-c',default=256,type=int,
        help='crop size of the hr image')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
        help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
        help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
        help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str,
        help='path to latest checkpoint (default: none)')
parser.add_argument('--generator',default='',type=str,
        help='path to the pretrained srResNet if use it')
parser.add_argument('--logdir','-s',default='save',type=str,
        help='path to save checkpoint')
parser.add_argument('--traindir',default='r375-400.bin',
        help=' the global name of training set dir')
parser.add_argument('--valdir',default='b100',type=str,
        help='the global name of validation set dir')
parser.add_argument('--weight',default=1,type=float,
        help='the weight of adversarial loss,i.e. gen_loss=content_loss+weight*adv_loss')
parser.add_argument('--fixG',action='store_true',
        help='wheather to fix generator and only train discriminator')
parser.add_argument('--fixD',action='store_true',
        help='wheather to fix discriminator and only train generator')
parser.add_argument('--clip',default=None,type=float,
        help='gradient clip norm')
parser.add_argument('--prefix',default='',type=str,
        help='prefix of the model name')

global_step=0
best_psnr = -100
best_gen_loss=10000
best_disc_loss= 10000
args = parser.parse_args()
args.__dict__['upscale_factor']=4
#args.traindir=globals()[args.traindir]
args.valdir=globals()[args.valdir]
args.__dict__['model_base_name']=args.prefix+'v%g_w%g'%(args.lr,args.weight)+('_fixG' if args.fixG else '')+('_fixD' if args.fixD else '')
args.__dict__['model_name']=args.model_base_name+'.pth'
args.__dict__['snapshot']='snapshot_'+args.model_base_name

setproctitle(args.model_base_name)
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

fclip=Clip().cuda()


if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        global_step=checkpoint['global_step']
        gen.load_state_dict(checkpoint['gen_state_dict'])
        disc.load_state_dict(checkpoint['disc_state_dict'])
        print("=> loaded checkpoint (epoch {})"
                .format( checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

if args.generator:
    if os.path.isfile(args.generator):
        print("=> loading checkpoint '{}'".format(args.generator))
        checkpoint = torch.load(args.generator)
        gen.load_state_dict(checkpoint['gen_state_dict'])
        print("=> loaded checkpoint")
    else:
        print("=>no checkpoint found at '{}'".format(args.generator))

cont_criterion = nn.MSELoss().cuda()
adv_criterion = nn.BCELoss().cuda()
real_label=Variable(torch.FloatTensor(torch.ones(args.batch_size)).cuda())
fake_label=Variable(torch.FloatTensor(torch.zeros(args.batch_size)).cuda())


gen_optimizer = torch.optim.Adam(gen.parameters(), args.lr)
disc_optimizer = torch.optim.Adam(disc.parameters(),args.lr)

def normalize(tensor):
    r,g,b=torch.split(tensor,1,1)
    r=(r-0.485)/0.229
    g=(g-0.456)/0.224
    b=(b-0.406)/0.225
    return torch.cat((r,g,b),1)

def rgb2y_matlab(img):   #convert a PIL rgb image to y-channel image in `matlab` way
    img=(img/255.0)[4:-4,4:-4,:]
    img=65.481*img[:,:,0]+128.553*img[:,:,1]+24.966*img[:,:,2]+16.0
    return img

def validate(model,vgg,criterion,valdir,epoch,factor,optimizer):
    cnt=0
    sum=0
    ysum=0
    cont_loss=0
    for x in [os.path.join(valdir,y) for y in os.listdir(valdir)]:
        im=Image.open(x)
        im=im.resize((im.size[0]-im.size[0]%factor,im.size[1]-im.size[1]%factor),Image.BICUBIC)
        target=Variable(torch.stack([transforms.ToTensor()(im)],0),volatile=True).cuda()
        #f_target=vgg(normalize(target))
        input=torch.stack([transforms.ToTensor()(im.resize((im.size[0]//args.upscale_factor,im.size[1]//args.upscale_factor),Image.BICUBIC))],0)
        input_var=Variable(input,volatile=True).cuda()
        output=model(input_var)
        #f_output=vgg(normalize(output))
        #cont_loss+=criterion(f_target,f_output).cpu().data[0]
        output=fclip(output)
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
    cont_loss=float(cont_loss)/cnt
    s=' | psnr=%.3f | ypsnr=%.3f '%(psnr,ypsnr)
    return s
    
def save_checkpoint(state, is_best,logdir):
    filename=os.path.join(logdir,args.model_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(logdir,'best_'+args.model_name))

def train(epoch):
    for i, (input, target) in enumerate(train_loader):
        global global_step
        if(global_step%4000==0):
            for pg in gen_optimizer.param_groups:
                pg['lr']=pg['lr']*0.915
            for pg in disc_optimizer.param_groups:
                pg['lr']=pg['lr']*0.915
        
        input_var = Variable(input.cuda())
        target_var = Variable(target.cuda())
        
        if not args.fixD:
            disc_optimizer.zero_grad()
            real_output=disc(target_var)
            real_loss=adv_criterion(real_output,real_label)
            real_loss.backward()
            disc_optimizer.step()

            disc_optimizer.zero_grad()
            fake_output=disc(gen(input_var).detach())
            fake_loss=adv_criterion(fake_output,fake_label)
            fake_loss.backward()
            disc_optimizer.step()
            disc_loss=fake_loss+real_loss
            
        if not args.fixG:
            gen_optimizer.zero_grad()
            G_z=gen(input_var)
            x1,x2,x3,x4,x5,x6=vgg(normalize(G_z))
            y1,y2,y3,y4,y5,y6=vgg(normalize(target_var))
            output=disc(G_z)
            adv_loss=adv_criterion(output,real_label)
            content_loss=\
                     cont_criterion(x2,y2.detach())\
                     +cont_criterion(x3,y3.detach())\
                     +cont_criterion(x4,y4.detach())\
                     +cont_criterion(x5,y5.detach())\
                     +cont_criterion(x6,y6.detach())\
                     +adv_loss*args.weight
            #content_loss=cont_criterion(fake_feature,real_feature)
            gen_loss=content_loss
            gen_loss.backward()
            gen_optimizer.step()
            
        if i % args.print_freq == 0:
            s=time.strftime('%dth-%H:%M:%S',time.localtime(time.time()))+' | epoch%d(%d) | lr=%g'%(epoch,global_step,args.lr)
            if not args.fixG:
                vs=validate(gen,vgg,cont_criterion,args.valdir,epoch,args.upscale_factor,gen_optimizer)
                s+=vs+' | Loss(G):%g[Cont:%g/Adv:%g]'%(gen_loss.data[0],content_loss.data[0],adv_loss.data[0])
            if not args.fixD:
                s+=' | Loss(D):%g[Real:%g/Fake:%g]'%(disc_loss.data[0],real_loss.data[0],fake_loss.data[0])
            print(s)
            f=open('info.'+args.model_base_name,'a')
            f.write(s+'\n')
            f.close()

            if not args.fixG:
                global best_gen_loss
                is_best = gen_loss.data[0] < best_gen_loss
                best_gen_loss = min(gen_loss.data[0] , best_gen_loss)
            else:
                global best_disc_loss
                is_best = disc_loss.data[0] < best_disc_loss
                best_disc_loss = min(disc_loss.data[0],best_disc_loss)
                
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step':global_step,
                'gen_state_dict': gen.state_dict(),
                'disc_state_dict':disc.state_dict(),
            }, is_best,args.logdir)

        global_step+=1

for epoch in range(args.start_epoch, args.epochs):
    train(epoch)
