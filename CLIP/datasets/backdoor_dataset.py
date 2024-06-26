import os
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np
import torch
import random

import copy


class SurrogateDataset(Dataset):
    """cifar10 dataset."""

    def __init__(self, img_dir, victim_feature_bank_dir, transform, view_transform, k):
        """
        """
        assert (transform is not None), "Transform can't be None"
        assert (view_transform is not None), "View transform can't be None"
        self.img_npz = np.load(img_dir)
        self.data = self.img_npz['x']
        self.transform = transform
        self.view_transform = view_transform
        self.query_num = self.data.shape[0]
        self.victim_feature_bank = np.load(victim_feature_bank_dir)['x']
        self.k = k

    def __len__(self):
        return self.query_num

    def __getitem__(self, index):
        # seed = index*10
        # random.seed(seed)
        # os.environ['PYTHONHASHSEED'] = str(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)

        # print("seed="+str(seed))
        # exit()

        img = self.data[index]
        img = Image.fromarray(img)
        img_raw = self.transform(img)
        views = [self.view_transform(img) for _ in range(self.k)]
        victim_feature = self.victim_feature_bank[index]
        return img_raw, views, victim_feature


class ReferenceImg(Dataset):

    def __init__(self, reference_file, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.target_input_array = np.load(reference_file)

        self.data = self.target_input_array['x']
        self.targets = self.target_input_array['y']

        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


class BadEncoderDataset(Dataset):
    """CIFAR10 Dataset.
    """

    def __init__(self, numpy_file, trigger_file, reference_file, indices, class_type, transform=None, bd_transform=None, ftt_transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']
        self.targets = self.input_array['y'][:,0].tolist()

        self.trigger_input_array = np.load(trigger_file)
        self.target_input_array = np.load(reference_file)

        self.trigger_patch_list = self.trigger_input_array['t']
        self.trigger_mask_list = self.trigger_input_array['tm']
        self.target_image_list = self.target_input_array['x']




        self.classes = class_type
        self.indices = indices
        self.transform = transform
        self.bd_transform = bd_transform
        self.ftt_transform = ftt_transform

    def __getitem__(self, index):
        img = self.data[self.indices[index]]
        img_copy = copy.deepcopy(img)
        backdoored_image = copy.deepcopy(img)
        img = Image.fromarray(img)
        '''original image'''
        if self.transform is not None:
            im_1 = self.transform(img)
        img_raw = self.bd_transform(img)
        '''generate backdoor image'''

        img_backdoor_list = []
        for i in range(len(self.target_image_list)):
            backdoored_image[:,:,:] = img_copy * self.trigger_mask_list[i] + self.trigger_patch_list[i][:]
            img_backdoor =self.bd_transform(Image.fromarray(backdoored_image))
            img_backdoor_list.append(img_backdoor)

        target_image_list_return, target_img_1_list_return = [], []
        for i in range(len(self.target_image_list)):
            target_img = Image.fromarray(self.target_image_list[i])
            target_image = self.bd_transform(target_img)
            target_img_1 = self.ftt_transform(target_img)
            target_image_list_return.append(target_image)
            target_img_1_list_return.append(target_img_1)

        return img_raw, img_backdoor_list, target_image_list_return, target_img_1_list_return
    def __len__(self):
        return len(self.indices)












class BadEncoderTestBackdoor(Dataset):
    """cifar10 dataset."""


    def __init__(self, numpy_file, trigger_file, reference_label, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']


        self.trigger_input_array = np.load(trigger_file)

        self.trigger_patch_list = self.trigger_input_array['t']
        self.trigger_mask_list = self.trigger_input_array['tm']

        self.target_class = reference_label

        self.test_transform = transform

    def __getitem__(self,index):
        img = copy.deepcopy(self.data[index])
        img[:] =img * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        img_backdoor =self.test_transform(Image.fromarray(img))
        return img_backdoor, self.target_class


    def __len__(self):
        return self.data.shape[0]



class CIFAR10CUSTOM(Dataset):
    """cifar10 dataset."""

    def __init__(self, numpy_file, class_type, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']
        try:
            self.targets = self.input_array['y'][:,0].tolist()
        except:
            self.targets = self.input_array['y']
        self.classes = class_type
        self.transform = transform
    def __len__(self):
        return self.data.shape[0]


class CIFAR10Pair(CIFAR10CUSTOM):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2


class CIFAR10Mem(CIFAR10CUSTOM):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target
