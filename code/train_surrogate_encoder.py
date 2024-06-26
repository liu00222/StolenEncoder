import os
import argparse
import torchvision
import numpy as np

from datetime import datetime
from functools import partial
from PIL import Image
from torch.utils.data import DataLoader
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

from evaluation import test
from datasets import get_surrogate_dataset
from models import get_surrogate_model


parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='resnet34', type=str, help='model architecture')
parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
parser.add_argument('--epochs', default=1000, type=int, help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', default='1', type=str, help='which gpu the code runs on')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--lambda_value', default=9.0, type=float)
parser.add_argument('--k', default=9, type=int)
parser.add_argument('--base_dataset', default='cifar10', type=str)
parser.add_argument('--base_model', default='simclr_clean_epoch_1000_trial_0', type=str)
parser.add_argument('--base_epoch', default='1000', type=str)
parser.add_argument('--surrogate_dataset', default='', type=str)
parser.add_argument('--distance', default='l2', type=str)
parser.add_argument('--seed', default=100, type=int, help='which gpu the code runs on')
args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


args.img_dir = f'../data/{args.base_dataset}_{args.base_model}_{args.base_epoch}/{args.surrogate_dataset}/train_img.npz'
args.victim_feature_bank_dir = f'../data/{args.base_dataset}_{args.base_model}_{args.base_epoch}/{args.surrogate_dataset}/train_feature.npz'

train_data = get_surrogate_dataset(args)

print(f'query number: {train_data.victim_feature_bank.shape[0]}')
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

model = get_surrogate_model(args).cuda()


def train(net, data_loader, train_optimizer, epoch, args):
    net.g.eval()
    net.f.train()

    # for module in net.f.modules():
    # # print(module)
    #     if isinstance(module, nn.BatchNorm2d):
    #         if hasattr(module, 'weight'):
    #             module.weight.requires_grad_(False)
    #         if hasattr(module, 'bias'):
    #             module.bias.requires_grad_(False)
    #         module.eval()
    #
    #
    # clean_net.eval()


    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)

    for img_raw, views, fv in train_bar:
        img_raw, fv = img_raw.cuda(non_blocking=True), fv.cuda(non_blocking=True)
        for i in range(len(views)):
            views[i] = views[i].cuda(non_blocking=True)

        loss = net(img_raw, views, fv, args.lambda_value, args.k, args.distance)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.6f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr'], total_loss / total_num))

    return total_loss / total_num


# define optimizer
print("Optimizer: SGD")
optimizer = torch.optim.SGD(model.f.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)

for param in model.g.parameters():
    param.requires_grad = False

results_dir = f'./output/{args.base_dataset}_{args.base_model}_{args.base_epoch}'
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

# adjust for the distance metric
if args.distance == 'l2':
    results_dir += f'/{args.surrogate_dataset}_lambda_{args.lambda_value}_k_{args.k}_arch_{args.arch}'
else:
    results_dir += f'/{args.surrogate_dataset}_lambda_{args.lambda_value}_k_{args.k}_arch_{args.arch}_distance_{args.distance}'

if not os.path.exists(results_dir):
    os.mkdir(results_dir)

# training loop
for epoch in range(1, args.epochs + 1):
    print("=================================================")
    train_loss = train(model, train_loader, optimizer, epoch, args)

    if epoch in [100, 300, 500, 700, 1000]:
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, results_dir + '/model_' + str(epoch) + '.pth')

    # torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, results_dir + '/model_last.pth')
