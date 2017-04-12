import argparse
import os
import shutil
import time
import numpy as np

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
from models import GenNet
from dataset import *
parser = argparse.ArgumentParser(description='Pretraining')
parser.add_argument('-j', '--workers', default=4, type=int,
        help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=5000, type=int,
        help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
        help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
        help='mini-batch size (default: 16)')
parser.add_argument('--crop-size','-c',default=128,type=int,
        help='crop size of the hr image')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
        help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
        help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
        help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str,
        help='path to latest checkpoint (default: none)')
parser.add_argument('--logdir','-s',default='save',type=str,
        help='path to save checkpoint')

best_psnr = -100
args = parser.parse_args()
args.__dict__['upscale_factor']=2
args.__dict__['train_filenames']='r128-256.bin'
args.__dict__['val_filenames']='test_r128-512.bin'

args.logdir='save'
if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)
cudnn.benchmark = True

train_loader = torch.utils.data.DataLoader(
    get_train_set(args),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    get_val_set(args),
    batch_size=args.batch_size, shuffle=True,
    num_workers=1, pin_memory=True)

model=GenNet().cuda()
#model = torch.nn.DataParallel(model).cuda()

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_psnr = checkpoint['best_psnr']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})"
                .format( checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

criterion = nn.MSELoss().cuda()

optimizer = torch.optim.Adam(model.parameters(), args.lr,weight_decay=args.weight_decay)

#if args.evaluate:
#    validate(val_loader, model, criterion)
#    return

val_iter=enumerate(val_loader)

def validate( model, criterion):
    global val_iter
    try:
        _, (input, target) = val_iter.next()
    except Exception,e:
        val_iter=enumerate(val_loader)
        _, (input, target) = val_iter.next()
    input_var = Variable(input.cuda(), volatile=True)
    target_var = Variable(target.cuda(), volatile=True)
    output = model(input_var)
    mse = criterion(output, target_var).cpu()
    psnr = 10*np.log10(1/mse.data[0])
    return psnr,'['+str(mse.data[0])+'/'+str(psnr)+']'

def save_checkpoint(state, is_best,logdir):
    filename=os.path.join(logdir,'model_bsd300_15.pth')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(logdir,'model_bsd300_15_best.pth'))

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    
    for i, (input, target) in enumerate(train_loader):
        input_var = Variable(input.cuda())
        target_var = Variable(target.cuda())

        output = model(input_var)
        loss = criterion(output, target_var)
        mse=loss.cpu()
        psnr = 10*np.log10(1/mse.data[0]) 

        optimizer.zero_grad()
        mse.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            bn_psnr,bn_s = validate( model, criterion)
            model.eval()
            nobn_psnr,nobn_s = validate( model,criterion) 
            model.train()
            psnr = max(bn_psnr,nobn_psnr)
            global best_psnr
            is_best = psnr > best_psnr
            best_psnr = max(psnr, best_psnr)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_psnr': best_psnr,
            }, is_best,args.logdir)
            s=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+': Epoch[{0}]({1}/{2}) '.format(epoch,i,len(train_loader))+' mse/psnr:'+'train['+str(mse.data[0])+'/'+str(psnr)+']'+' bnEval'+bn_s+' nobnEval'+nobn_s  
            f=open('info.pretrain_bsd300_15','a')
            f.write(s+'\n')
            f.close()
            print(s)

for epoch in range(args.start_epoch, args.epochs):
    train(train_loader, model, criterion, optimizer, epoch)
