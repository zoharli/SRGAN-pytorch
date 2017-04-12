from torchvision.transforms import Compose, RandomCrop, ToTensor, Scale, RandomHorizontalFlip
from torch.utils.data import Dataset
from os.path import join
from PIL import Image
import os 
import sys
path='/media/H/zohar/SRGAN-pytorch/BSDS300-images/BSDS300/images/'

def transform_target(crop_size):
    """Ground truth image    """
    return Compose([
        RandomCrop(crop_size),
        RandomHorizontalFlip(),
        ])


def transform_input(crop_size, upscale_factor):
    """LR of target image    """
    return Compose([
        Scale(crop_size // upscale_factor**2,3),
        ])


def fit_crop_size(crop_size, upscale_factor):
    return crop_size - crop_size % upscale_factor


def get_train_set(args):
    crop_size = args.crop_size
    crop_size = fit_crop_size(crop_size, args.upscale_factor)
    return ImageDataset(
        path+'train',
        transform_target=transform_target(crop_size),
        transform_input=transform_input(crop_size, args.upscale_factor))


def get_val_set(args):
    crop_size = args.crop_size
    crop_size = fit_crop_size(crop_size, args.upscale_factor)
    return ImageDataset(
        path+'test',
        transform_target=transform_target(crop_size),
        transform_input=transform_input(crop_size, args.upscale_factor))

class ImageDataset(Dataset):
    def __init__(self,path, transform_target=None, transform_input=None):
        super(ImageDataset,self).__init__()
        self.image_filenames = [os.path.join(path,x) for x in os.listdir(path)]
        self.transform_target = transform_target
        self.transform_input = transform_input

    def __getitem__(self, index):
        target = Image.open(self.image_filenames[index])
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

