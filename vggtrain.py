import time
import sys
import os
import torch
import torchvision
import argparse
from vgg import *
from models import *
from dataset import *
from setproctitle import *
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

parser=argparse.ArgumentParser(description='this script is wrote for fine tunning vgg before trainning SRGAN')
parser.add_argument('-j', '--workers', default=4, type=int,
        help='number of data loading workers (default: 4)')
parser.add_argument('--batch-size','-b',default=8)
parser.add_argument('--epochs', default=40, type=int,
        help='number of total epochs to run')
parser.add_argument('--crop-size','-c',default=256,type=int,
        help='crop size of the image')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
        help='initial learning rate')
parser.add_argument('--print-freq', '-p', default=100, type=int,
        help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str,
        help='path to latest checkpoint (default: none)')
parser.add_argument('--logdir','-s',default='save',type=str,
        help='path to save checkpoint')
parser.add_argument('--traindir',default='r375-400.bin',
        help=' the global name of training set dir')
parser.add_argument('--generator', default=None,type=str,
        help='path to generator for producing fake image')
parser.add_argument('--fixF',action='store_true')

global_step=1
args = parser.parse_args()
args.__dict__['upscale_factor']=4
#args.traindir=globals()[args.traindir]
args.__dict__['model_base_name']='vgg_v%g'%(args.lr)+('_fixF' if args.fixF else '')
args.__dict__['model_name']=args.model_base_name+'.pth'

setproctitle(args.model_base_name)
if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)

cudnn.benchmark = True

train_loader = torch.utils.data.DataLoader(
    get_train_set(args),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)

gen=GenNet().cuda()
vgg=vgg19_54().cuda()

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        global_step=checkpoint['global_step']
        vgg.load_state_dict(checkpoint['vgg_state_dict'])
        print("=> loaded checkpoint ")
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

if os.path.isfile(args.generator):
    print("=> loading checkpoint '{}'".format(args.generator))
    checkpoint = torch.load(args.generator)
    gen.load_state_dict(checkpoint['gen_state_dict'])
    print("=> loaded checkpoint")
else:
    print("=>no generator checkpoint found at '{}'".format(args.generator))
    sys.exit()

if args.fixF:
    vgg_optimizer = torch.optim.Adam(vgg.classifier.parameters(),args.lr)
else:    
    vgg_optimizer = torch.optim.Adam(vgg.parameters(),args.lr)
cri=torch.nn.SoftMarginLoss().cuda()

def normalize(tensor):
    r,g,b=torch.split(tensor,1,1)
    r=(r-0.485)/0.229
    g=(g-0.456)/0.224
    b=(b-0.406)/0.225
    return torch.cat((r,g,b),1)

def save_checkpoint(state, logdir):
    filename=os.path.join(logdir,args.model_name)
    torch.save(state, filename)

real_label=Variable(torch.FloatTensor(torch.ones(args.batch_size)).cuda())
fake_label=Variable(torch.FloatTensor(-1*torch.ones(args.batch_size)).cuda())

def train(epoch):
    for i, (input, target) in enumerate(train_loader):
        global global_step
        if(global_step%1000==0):
            for pg in vgg_optimizer.param_groups:
                pg['lr']=pg['lr']*0.915
        
        input_var = Variable(input.cuda())
        target_var = Variable(target.cuda())
        
        vgg_optimizer.zero_grad()
        t=vgg(normalize(target_var))[-1]
        rloss=cri(t,real_label)
        rloss.backward()
        output = gen(input_var).detach()
        o=vgg(normalize(output))[-1]
        floss=cri(o,fake_label)
        floss.backward()
        loss=floss+rloss
        vgg_optimizer.step()    
            
        if i % args.print_freq == 0:
            s=time.strftime('%dth-%H:%M:%S',time.localtime(time.time()))+' | epoch%d(%d) | lr=%g'%(epoch,global_step,args.lr)+' | loss=%g[r:%g/f:%g]'%(loss.cpu().data[0],rloss.cpu().data[0],floss.cpu().data[0])
            print(s)
            f=open('info.'+args.model_base_name,'a')
            f.write(s+'\n')
            f.close()
            save_checkpoint({
                'global_step':global_step,
                'vgg_state_dict': vgg.state_dict(),
            }, args.logdir)

        global_step+=1

for epoch in range(args.epochs):
    train(epoch)
