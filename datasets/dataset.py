from .augmentation.randaugment import RandAugment  # Assuming this is adapted for PyTorch

from torch.utils.data import Dataset  # Import PyTorch's Dataset
from torchvision import transforms as T  # Import PyTorch transforms
from PIL import Image
import numpy as np
import copy

mean, std = {}, {}
mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
mean['svhn'] = [0.4380, 0.4440, 0.4730]
mean['stl10'] = [x / 255 for x in [112.4, 109.1, 98.6]]
mean['imagenet'] = [0.485, 0.456, 0.406]

std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]
std['svhn'] = [0.1751, 0.1771, 0.1744]
std['stl10'] = [x / 255 for x in [68.4, 66.6, 68.5]]
std['imagenet'] = [0.229, 0.224, 0.225] #rgb

class BasicDataset(Dataset):

    def __init__(self, name, data,targets=None,num_classes=None,transform=None, is_ulb=False, strong_transform=None, *args, **kwargs):
        super(BasicDataset, self).__init__()
        self.name = name
        self.data = data
        self.targets = targets

        self.num_classes = num_classes
        self.is_ulb = is_ulb

        self.transform = transform
        if self.is_ulb:
            self.strong_transform = RandAugment()
        else:
            self.strong_transform = strong_transform
        assert self.name in mean, 'please check datset name'

    def __getitem__(self, idx):
          target = self.targets[idx]
          img = self.data[idx]  # Assuming data is a list of PIL Images or file paths

          if self.transform is None:
              img = self.standard_trans(img)
              return idx, img, target
          else:
            # Apply weak augmentation
              img_w = self.transform(img)  # Apply weak augmentation (assuming it's a PyTorch transform)
              img_w = self.standard_trans(img_w) 

          if not self.is_ulb:
              return idx, img_w, target
          else:
              assert self.strong_transform is not None, 'unlabeled data needs strong_transform'
              # Apply strong augmentation for unlabeled data
              img_s = self.transform(img)  # Apply weak augmentation first
              img_s = self.strong_transform(img_s)  # Apply strong augmentation
              img_s = self.standard_trans(img_s)  # Apply standard transformations
              return idx, img_w, img_s, target  # Return idx, img_w, img_s, and target for unlabeled data

    def __len__(self):
        return len(self.data)

    def standard_trans(self, img):
        assert len(img.shape) == 3, 'img shape must be 3'
        img = np.ascontiguousarray(np.rollaxis(img, 2))/255.
        img = (img - np.array(mean[self.name]).reshape(-1,1,1)) / np.array(std[self.name]).reshape(-1,1,1)
        return img
