import os
import argparse
import random

import torchvision
import numpy as np
from functools import partial
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import get_encoder_architecture
from evaluation import create_torch_dataloader, NeuralNet, net_train, net_test, predict_feature
from datasets import get_custom_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--pretraining_dataset', default='CLIP', type=str)
parser.add_argument('--clip_dir', default='', type=str)
parser.add_argument('--surrogate_dataset', default='stl10_unlabel', type=str)
parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--seed', default=100, type=int)
parser.add_argument('--gpu', default='0', type=str)
args = parser.parse_args()  # running in command line

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
print(f'Loading from: {args.clip_dir}')
model = get_encoder_architecture(args).cuda()
checkpoint = torch.load(args.clip_dir)
model.visual.load_state_dict(checkpoint['state_dict'])

memory_data = get_custom_dataset(f'/path/to/{args.surrogate_dataset}/train_224.npz', 'CLIP')
test_data = get_custom_dataset(f'/path/to/{args.surrogate_dataset}/test_224.npz', 'CLIP')
train_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
feature_bank_training, label_bank_training = predict_feature(model.visual, train_loader)
feature_bank_testing, label_bank_testing = predict_feature(model.visual, test_loader)

print(feature_bank_training.shape)
print(feature_bank_testing.shape)

feature_bank_mix = np.concatenate((feature_bank_training, feature_bank_testing), axis=0)

print(feature_bank_mix.shape)

save_dir = f'../data/CLIP/{args.surrogate_dataset}'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

np.savez(f'{save_dir}/queries.npz', x=feature_bank_mix)
print(f'Queried feature vectors are saved at {save_dir}/queries.npz\n')

train_x = np.load(f'/path/to/{args.surrogate_dataset}/train_224.npz')['x']
test_x = np.load(f'/path/to/{args.surrogate_dataset}/test_224.npz')['x']
data_mix = np.concatenate((train_x, test_x), axis=0)
print(data_mix.shape)
np.savez(f'{save_dir}/data.npz', x=data_mix)
print(f'Images are saved at {save_dir}/data.npz\n')
