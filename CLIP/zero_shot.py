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

from evaluation import create_torch_dataloader, NeuralNet, net_train, net_test, predict_feature
from datasets import get_custom_dataset
from models import get_encoder_architecture, get_CLIP_surrogate


parser = argparse.ArgumentParser()

parser.add_argument('--arch', default='CLIP', type=str, help='model architecture')
parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
parser.add_argument('--epochs', default='1000', type=str, help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', default='1', type=str, help='which gpu the code runs on')

parser.add_argument('--pretraining_dataset', default='CLIP', type=str)
parser.add_argument('--surrogate_dataset', default='stl10_unlabel', type=str)

parser.add_argument('--sim_score_only', action='store_true')
parser.add_argument('--downstream_dataset', default='cifar10', type=str)
parser.add_argument('--seed', default=100, type=int, help='which gpu the code runs on')
parser.add_argument('--classifier', default='network', type=str)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--nn_epochs', default=500, type=int)
parser.add_argument('--hidden_size_1', default=512, type=int)
parser.add_argument('--hidden_size_2', default=256, type=int)

parser.add_argument('--lambda_value', default=9.0, type=float)
parser.add_argument('--k', default=9, type=int)
parser.add_argument('--use_exp', action='store_true')
args = parser.parse_args()  # running in command line

# args = parser.parse_args('')  # running in ipynb


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

gtsrb_classes = ['Speed limit (20km/h)',
                'Speed limit (30km/h)',
                'Speed limit (50km/h)',
                'Speed limit (60km/h)',
                'Speed limit (70km/h)',
                'Speed limit (80km/h)',
                'End of speed limit (80km/h)',
                'Speed limit (100km/h)',
                'Speed limit (120km/h)',
                'No passing',
                'No passing for vehicles over 3.5 metric tons',
                'Right-of-way at the next intersection',
                'Priority road',
                'Yield',
                'Stop',
                'No vehicles',
                'Vehicles over 3.5 metric tons prohibited',
                'No entry',
                'General caution',
                'Dangerous curve to the left',
                'Dangerous curve to the right',
                'Double curve',
                'Bumpy road',
                'Slippery road',
                'Road narrows on the right',
                'Road work',
                'Traffic signals',
                'Pedestrians',
                'Children crossing',
                'Bicycles crossing',
                'Beware of ice/snow',
                'Wild animals crossing',
                'End of all speed and passing limits',
                'Turn right ahead',
                'Turn left ahead',
                'Ahead only',
                'Go straight or right',
                'Go straight or left',
                'Keep right',
                'Keep left',
                'Roundabout mandatory',
                'End of no passing',
                'End of no passing by vehicles over 3.5 metric tons']

num_of_classes = 10
if args.downstream_dataset == 'gtsrb':
    num_of_classes = 43
elif args.downstream_dataset == 'cifar10':
    num_of_classes = 10
elif args.downstream_dataset == 'cifar100':
    num_of_classes = 100
elif args.downstream_dataset == 'svhn':
    num_of_classes = 10
elif args.downstream_dataset == 'stl10':
    num_of_classes = 10
elif args.downstream_dataset == 'resisc45':
    num_of_classes = 45
elif args.downstream_dataset == 'dtd':
    num_of_classes = 47
elif args.downstream_dataset == 'food101':
    num_of_classes = 101
elif args.downstream_dataset == 'mnist':
    num_of_classes = 10
elif args.downstream_dataset == 'fashion_mnist':
    num_of_classes = 10
elif args.downstream_dataset == 'eurosat':
    num_of_classes = 10
else:
    raise NotImplementedError

print(f'\n{args.downstream_dataset} classes: {num_of_classes}')

import clip

victim, _ = clip.load('RN50', 'cuda')

if args.use_exp:
    print('loading the model with expectation loss')
    surrogate_dir = f'./output/{args.arch}_{args.surrogate_dataset}_lambda_{args.lambda_value}_k_{args.k}_exp/model_{args.epochs}.pth'
else:
    surrogate_dir = f'./output/{args.arch}_{args.surrogate_dataset}_lambda_{args.lambda_value}_k_{args.k}/model_{args.epochs}.pth'
print(f'\nLoading the surrogate encoder from {surrogate_dir}')
surrogate = get_CLIP_surrogate(args).cuda()
surrogate.load_state_dict(torch.load(surrogate_dir)['state_dict'])


print(f'\nLoading the downstream dataset {args.downstream_dataset}')

victim_test = get_custom_dataset(f'/path/to/{args.downstream_dataset}/test_224.npz', args.pretraining_dataset)

surrogate_test = get_custom_dataset(f'/path/to/{args.downstream_dataset}/test_224.npz', None)

empty_transform = transforms.Compose([
    transforms.ToTensor()])

test_transform_CLIP = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])



def eval(model, test_data, preprocess):
    # print(np.min(test_data.targets))
    # print(np.max(test_data.targets))
    # exit()
    if args.downstream_dataset == 'gtsrb':
        print('loading from gtsrb')
        # text_inputs = torch.cat([clip.tokenize(f"A traffic sign photo of a {c}") for c in gtsrb_classes]).to('cuda')
        text_inputs = torch.cat([clip.tokenize(f"A photo of a {c}") for c in gtsrb_classes]).to('cuda')
    elif args.downstream_dataset == 'mnist':
        print('loading from mnist')
        text_inputs = torch.cat([clip.tokenize(f"A photo of a {c}") for c in ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']]).to('cuda')
        # digit photo of
    elif args.downstream_dataset == 'fashion_mnist':
        print('loading from fashion_mnist')
        text_inputs = torch.cat([clip.tokenize(f"A photo of a {c}") for c in ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']]).to('cuda')
    elif args.downstream_dataset == 'svhn':
        print('loading from svhn')
        text_inputs = torch.cat([clip.tokenize(f"A photo of a {c}") for c in ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']]).to('cuda')
    elif args.downstream_dataset == 'eurosat':
        print('loading from eurosat')
        text_inputs = torch.cat([clip.tokenize(f"A photo of a {c}") for c in ['AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial','Pasture','PermanentCrop','Residential','River','SeaLake']]).to('cuda')
    else:
        raise NotImplementedError

    with torch.no_grad():
        text_features = victim.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    hit = 0

    total_num = test_data.data.shape[0]

    for i in tqdm(range(total_num)):
        # Prepare the inputs
        image, class_id = Image.fromarray(test_data.data[i]), test_data.targets[i]
        image_input = preprocess(image).unsqueeze(0).to('cuda')

        #text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.visual(image_input)
            #text_features = model.encode_text(text_inputs)

        # Pick the top 1 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.to(torch.float16)
        # print(text_features)
        # print(image_features)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)

        # Print the result
        # print("\nTop predictions:\n")
        # for value, index in zip(values, indices):
        #     print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")

        if int(class_id) == int(indices.item()):
            hit += 1

    print(f"Accuracy: {float(hit) / total_num}")
    print()


eval(surrogate, surrogate_test, empty_transform)
eval(victim, victim_test, test_transform_CLIP)
