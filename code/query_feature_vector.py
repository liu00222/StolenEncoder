import os
import argparse
import torchvision
import numpy as np

from datetime import datetime
from functools import partial
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet
from tqdm import tqdm

import json
import math
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from nn_classifier import create_torch_dataloader
from datasets import get_custom_dataset
from models import get_model_clean

parser = argparse.ArgumentParser()

parser.add_argument('-a', '--arch', default='resnet18')
parser.add_argument('--feature_dim', default=128, type=int)

parser.add_argument('--base_dataset', default='cifar10', type=str)
parser.add_argument('--base_model', default='simclr_clean_epoch_1000_trial_0', type=str)
parser.add_argument('--base_epoch', default='1000', type=str)
parser.add_argument('--surrogate_dataset', default='', type=str)
parser.add_argument('--train', default='True', type=str)
parser.add_argument('--query_num', default=100000, type=int, help='only effective when surrogate_dataset is set to be random')

parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--seed', default=100, type=int)
parser.add_argument('--gpu', default='0', type=str)
args = parser.parse_args()

assert (args.surrogate_dataset), 'please specify the surrogate dataset name'

# set seed and gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# load the clean (victim) encoder
args.resume = f"../models/{args.base_dataset}/{args.base_model}/model_{args.base_epoch}.pth"
print('Loading clean encoder')
model = get_model_clean(args).cuda()


def predict_feature(net, data_loader):
    """
    Encode the data to a feature vector and a label vector.

    :type net: nn.Module
    :type data_loader: Dataloader
    :rtype: np.array, np.array
    """
    net.eval()
    # classes = len(data_loader.dataset.classes)
    feature_bank, target_bank = [], []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(data_loader, desc='Feature extracting'):
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            target_bank.append(target)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        target_bank = torch.cat(target_bank, dim=0).contiguous()

    return feature_bank.cpu().detach().numpy(), target_bank.detach().numpy()


print(f'Loading from: {args.resume}')
checkpoint = torch.load(args.resume)
model.load_state_dict(checkpoint['state_dict'])

# if the surrogate dataset is random, we generate the images randomly first
if args.surrogate_dataset == 'random':
    raise NotImplementedError

else:
    if args.train != 'True':
        original_data_npz = np.load(f'/path/to/{args.surrogate_dataset}/test.npz')
    else:
        original_data_npz = np.load(f'/path/to/{args.surrogate_dataset}/train.npz')

    train_x = original_data_npz['x']
    train_y = original_data_npz['y']
    print(train_x.shape)
    print(train_y.shape)

    print('number of query num:')
    print(args.query_num)
    if args.query_num > train_x.shape[0]:
        raise ValueError(f"Query number can't exceed the original training set size, which is {train_x.shape[0]}")
    training_data_sampling_indices = np.random.choice(train_x.shape[0], args.query_num, replace=False)
    train_x = train_x[training_data_sampling_indices]
    train_y = train_y[training_data_sampling_indices]
    print(train_x.shape)
    print(train_y.shape)


if args.train != 'True':
    args.surrogate_dataset = f'{args.surrogate_dataset}_test_num_{args.query_num}_seed_{args.seed}'
else:
    args.surrogate_dataset = f'{args.surrogate_dataset}_num_{args.query_num}_seed_{args.seed}'

img_save_dir = f'../data/{args.base_dataset}_{args.base_model}_{args.base_epoch}'
if not os.path.exists(img_save_dir):
    os.mkdir(img_save_dir)
img_save_dir += f'/{args.surrogate_dataset}'
if not os.path.exists(img_save_dir):
    os.mkdir(img_save_dir)
np.savez(f'{img_save_dir}/train_img.npz', x=train_x, y=train_y)

print(f'Images are saved at {img_save_dir}')

memory_data = get_custom_dataset(f'{img_save_dir}/train_img.npz', args.base_dataset)
train_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
feature_bank_training, label_bank_training = predict_feature(model.f, train_loader)

print(feature_bank_training.shape)
print(label_bank_training.shape)

save_dir = f'../data/{args.base_dataset}_{args.base_model}_{args.base_epoch}'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_dir += f'/{args.surrogate_dataset}'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

np.savez(f'{save_dir}/train_feature.npz', x=feature_bank_training, y=label_bank_training)
print(f'Training feature vectors are saved at {save_dir}/train.npz\n')
