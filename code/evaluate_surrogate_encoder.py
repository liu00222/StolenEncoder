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

parser = argparse.ArgumentParser()

parser.add_argument('--arch', default='resnet18', type=str, help='model architecture')
parser.add_argument('--archv', default='resnet18', type=str, help='model architecture')
parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
parser.add_argument('--epochs', default=1000, type=int, help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', default='1', type=str, help='which gpu the code runs on')
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--schedule', default=[], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')

parser.add_argument('--base_dataset', default='cifar10', type=str)
parser.add_argument('--base_model', default='simclr_clean_epoch_1000_trial_0', type=str)
parser.add_argument('--base_epoch', default='1000', type=str)
parser.add_argument('--surrogate_epoch', default='last', type=str)
parser.add_argument('--surrogate_dataset', default='', type=str)

parser.add_argument('--seed', default=100, type=int, help='which gpu the code runs on')

parser.add_argument('--fraction', default=100, type=int, help='')
parser.add_argument('--sim_score_only', action='store_true')
parser.add_argument('--downstream_dataset', default='cifar10', type=str)
parser.add_argument('--downstream_seed', default=100, type=int, help='which gpu the code runs on')
parser.add_argument('--classifier', default='network', type=str)
parser.add_argument('--distance', default='l2', type=str)
parser.add_argument('--nn_lr', default=0.0001, type=float)
parser.add_argument('--nn_epochs', default=500, type=int)
parser.add_argument('--hidden_size_1', default=512, type=int)
parser.add_argument('--hidden_size_2', default=256, type=int)

parser.add_argument('--lambda_value', default=9.0, type=float)
parser.add_argument('--k', default=9, type=int)
args = parser.parse_args()  # running in command line

# args = parser.parse_args('')  # running in ipynb


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

random.seed(args.downstream_seed)
os.environ['PYTHONHASHSEED'] = str(args.downstream_seed)
np.random.seed(args.downstream_seed)
torch.manual_seed(args.downstream_seed)
torch.cuda.manual_seed(args.downstream_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


from models import get_model_clean, get_surrogate_model

# save the arch of the clone model into archc
archc = args.arch

# archv is the arch of the victim model
args.arch = args.archv
victim_dir = f"/path/to/{args.base_dataset}/{args.base_model}/model_{args.base_epoch}.pth"
print(f'Loading the victim encoder from {victim_dir}')
victim = get_model_clean(args).cuda()
victim.load_state_dict(torch.load(victim_dir)['state_dict'])

args.arch = archc
if args.distance == 'l2':
    surrogate_dir = f'./output/{args.base_dataset}_{args.base_model}_{args.base_epoch}/{args.surrogate_dataset}_lambda_{args.lambda_value}_k_{args.k}_arch_{args.arch}/model_{args.surrogate_epoch}.pth'
elif args.distance == 'l1':
    surrogate_dir = f'./output/{args.base_dataset}_{args.base_model}_{args.base_epoch}/{args.surrogate_dataset}_lambda_{args.lambda_value}_k_{args.k}_arch_{args.arch}_distance_l1/model_{args.surrogate_epoch}.pth'
elif args.distance == 'cosine':
    surrogate_dir = f'./output/{args.base_dataset}_{args.base_model}_{args.base_epoch}/{args.surrogate_dataset}_lambda_{args.lambda_value}_k_{args.k}_arch_{args.arch}_distance_cosine/model_{args.surrogate_epoch}.pth'
else:
    raise NotImplementedError

print(f'\nLoading the surrogate encoder from {surrogate_dir}')
surrogate = get_surrogate_model(args).cuda()
surrogate.load_state_dict(torch.load(surrogate_dir)['state_dict'])


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

print(f'\nLoading the downstream dataset {args.downstream_dataset}')
from datasets import get_custom_dataset

victim_train = get_custom_dataset(f'/path/to/{args.downstream_dataset}/train.npz', args.base_dataset)
victim_test = get_custom_dataset(f'/path/to/{args.downstream_dataset}/test.npz', args.base_dataset)
victim_train_loader = DataLoader(victim_train, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
victim_test_loader = DataLoader(victim_test, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

num_of_classes = 10
if args.downstream_dataset == 'gtsrb':
    num_of_classes = 43
elif args.downstream_dataset == 'cifar10':
    num_of_classes = 10
elif args.downstream_dataset == 'cifar100':
    num_of_classes = 100
elif args.downstream_dataset == 'svhn':
    num_of_classes = 10
elif args.downstream_dataset == 'dtd':
    num_of_classes = 47
elif args.downstream_dataset == 'resisc45':
    num_of_classes = 45
elif args.downstream_dataset == 'flower':
    num_of_classes = 5
elif args.downstream_dataset == 'food101':
    num_of_classes = 101
elif args.downstream_dataset == 'stl10':
    num_of_classes = 10
elif args.downstream_dataset == 'mnist':
    num_of_classes = 10
elif args.downstream_dataset == 'fashion_mnist':
    num_of_classes = 10
elif args.downstream_dataset == 'eurosat':
    num_of_classes = 10
elif args.downstream_dataset == 'tinyimagenet':
    num_of_classes = 200
else:
    raise NotImplementedError

print(f'\n{args.downstream_dataset} classes: {num_of_classes}')

surrogate_train = get_custom_dataset(f'/path/to/{args.downstream_dataset}/train.npz', None)
surrogate_test = get_custom_dataset(f'/path/to/{args.downstream_dataset}/test.npz', None)
surrogate_train_loader = DataLoader(surrogate_train, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
surrogate_test_loader = DataLoader(surrogate_test, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

feature_bank_training_v, label_bank_training_v = predict_feature(victim.f, victim_train_loader)
feature_bank_testing_v, label_bank_testing_v = predict_feature(victim.f, victim_test_loader)
feature_bank_training_s, label_bank_training_s = predict_feature(surrogate.f, surrogate_train_loader)
feature_bank_testing_s, label_bank_testing_s = predict_feature(surrogate.f, surrogate_test_loader)

print(feature_bank_training_v.shape)
print(label_bank_training_v.shape)
print(feature_bank_testing_v.shape)
print(label_bank_testing_v.shape)

print(feature_bank_training_s.shape)
print(label_bank_training_s.shape)
print(feature_bank_testing_s.shape)
print(label_bank_testing_s.shape)


def AverageSim(feature_bank1, feature_bank2):
    sim_score = 0.0
    for i in range(feature_bank1.shape[0]):
        feature1, feature2 = feature_bank1[i], feature_bank2[i]
        similarity_score = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
        sim_score += similarity_score
    return sim_score / feature_bank1.shape[0]

sim_score_avg_clean_test = AverageSim(np.concatenate((feature_bank_training_v, feature_bank_testing_v), axis=0),
                                      np.concatenate((feature_bank_training_s, feature_bank_testing_s), axis=0))
print(f'The average cosine similarity between the features generated by the victim and the surrogate on {args.downstream_dataset} is: \n{sim_score_avg_clean_test}')

if args.sim_score_only:
    exit()


def downstream_classifier_predict(input_size, train_loader, test_loader, nn_path, keyword):
    net = NeuralNet(input_size, [args.hidden_size_1, args.hidden_size_2], num_of_classes).cuda()
    try:
        print(f'Trying to load the classifier from {nn_path}')
        net.load_state_dict(torch.load(f'{nn_path}/model_{args.nn_epochs}.pth')['state_dict'])
    except FileNotFoundError:
        print(f'{keyword} classifider not found')
        print('\nStart to train the classifider')
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=args.nn_lr)

        for epoch in range(1, args.nn_epochs + 1):
            net_train(net, train_loader, optimizer, epoch, criterion)
            net_test(net, test_loader, epoch, criterion)

        torch.save({'epoch': args.epochs, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}, f"{nn_path}/model_{args.nn_epochs}.pth")
        print(f"The {keyword} classifier is saved at: {nn_path}/model_{args.nn_epochs}.pth")

    prediction = np.array([])
    net.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            output = net(data)
            pred = output.argmax(dim=1, keepdim=True)
            prediction = np.append(prediction, pred.cpu().numpy().reshape(1,-1)[0])
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_acc = 100. * correct / len(test_loader.dataset)
    print(f'{keyword} classifier statistics: ')
    print('{{"metric": "Eval - Accuracy", "value": {}}}'.format(100. * correct / len(test_loader.dataset)))
    return prediction



if args.classifier == 'network':
    from nn_classifier import NeuralNet
    from nn_classifier import net_train, net_test

    # create the data loader for the victim classifier
    nn_train_loader_v = create_torch_dataloader(feature_bank_training_v, label_bank_training_v, args.batch_size)
    nn_test_loader_v = create_torch_dataloader(feature_bank_testing_v, label_bank_testing_v, args.batch_size)

    # create the data loader for the surrogate classifier
    nn_train_loader_s = create_torch_dataloader(feature_bank_training_s, label_bank_training_s, args.batch_size)
    nn_test_loader_s = create_torch_dataloader(feature_bank_testing_s, label_bank_testing_s, args.batch_size)

    input_size = feature_bank_training_v.shape[1]

    # create the directories to save the downstream classifiers (if not exist)
    victim_nn_path = f'./downstream_checkpoint/victim/{args.base_dataset}_{args.base_model}_{args.base_epoch}_{args.downstream_dataset}'
    if not os.path.exists(victim_nn_path):
        os.mkdir(victim_nn_path)
    surrogate_nn_path = f'./downstream_checkpoint/surrogate/{args.base_dataset}_{args.base_model}_{args.base_epoch}'
    if not os.path.exists(surrogate_nn_path):
        os.mkdir(surrogate_nn_path)
    surrogate_nn_path += f'/{args.surrogate_dataset}_lambda_{args.lambda_value}_k_{args.k}_surrogate_epochs_{args.surrogate_epoch}_arch_{args.arch}'
    if args.distance != 'l2':
        surrogate_nn_path += f'_distance_{args.distance}'
    if args.seed != 'l2':
        surrogate_nn_path += f'_downstream_seed_{args.downstream_seed}'
    if not os.path.exists(surrogate_nn_path):
        os.mkdir(surrogate_nn_path)
    surrogate_nn_path += f'/{args.downstream_dataset}'
    if not os.path.exists(surrogate_nn_path):
        os.mkdir(surrogate_nn_path)

    # train/load the downstream classifiers and obtain the predictions for the testing data
    victim_prediction = downstream_classifier_predict(input_size, nn_train_loader_v, nn_test_loader_v, victim_nn_path, 'Victim')
    surrogate_prediction = downstream_classifier_predict(input_size, nn_train_loader_s, nn_test_loader_s, surrogate_nn_path, 'Surrogate')
    agree = 0
    for i in range(victim_prediction.shape[0]):
        if victim_prediction[i] == surrogate_prediction[i]:
            agree += 1
    print(f'agreement: {agree / victim_prediction.shape[0]}')

else:
    raise NotImplementedError
