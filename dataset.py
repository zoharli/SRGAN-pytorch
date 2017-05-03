from torchvision.transforms import Compose, RandomCrop, ToTensor, Scale, RandomHorizontalFlip
from torch.utils.data import Dataset
import os 
from PIL import Image

import sys


datasets_root='/home/zohar/datasets/'
s291=datasets_root+'291'
s91=datasets_root+'91'
s14=datasets_root+'Set14'
s5=datasets_root+'Set5'
b200=datasets_root+'BSD200'
b100=datasets_root+'BSD100'

def transform_target(crop_size):
    """Ground truth image    """
    return Compose([
        RandomCrop(crop_size),
        RandomHorizontalFlip(),
        ])


def transform_input(crop_size, upscale_factor):
    """LR of target image    """
    return Compose([
        Scale(crop_size // upscale_factor,Image.BICUBIC),
        ])


def fit_crop_size(crop_size, upscale_factor):
    return crop_size - crop_size % upscale_factor


def get_train_set(args):
    crop_size = args.crop_size
    crop_size = fit_crop_size(crop_size, args.upscale_factor)
    return ImageDataset(
        args.traindir,
        args.crop_size,
        transform_target=transform_target(crop_size),
        transform_input=transform_input(crop_size, args.upscale_factor))


def get_val_set(args):
    crop_size = args.crop_size
    crop_size = fit_crop_size(crop_size, args.upscale_factor)
    return ImageDataset(
        args.valdir,
        transform_target=transform_target(crop_size),
        transform_input=transform_input(crop_size, args.upscale_factor))

class ImageDataset(Dataset):
    def __init__(self,path_file,crop_size, transform_target=None, transform_input=None):
        super(ImageDataset,self).__init__()
        #self.image_filenames = [os.path.join(path_file,x) for x in os.listdir(path_file)]
        self.image_filenames = open(path_file).read().split('\n')
        self.transform_target = transform_target
        self.transform_input = transform_input
        self.crop_size=crop_size

    def __getitem__(self, index):
        target = Image.open(self.image_filenames[index])
       # size=target.size
       # rate = self.crop_size*1.25/min(size)
       # target=target.resize((int(size[0]*rate),int(size[1]*rate)),Image.BICUBIC)
        if self.transform_target:
            target = self.transform_target(target)
        input = target.copy()
        if self.transform_input:
            input = self.transform_input(input)
        target = ToTensor()(target)
        input = ToTensor()(input)
        return input,target

    def __len__(self):
        return len(self.image_filenames)

