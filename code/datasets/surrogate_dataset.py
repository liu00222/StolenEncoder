from torchvision import transforms
from .backdoor_dataset import CIFAR10PairBD, CIFAR10PairBDFTT, CIFAR10Mem, CIFAR10Pair, CIFAR10TestBackdoor, CIFAR10TargetImage
from .backdoor_dataset import CIFAR10PairBDMulti, SurrogateDataset
import numpy as np


finetune_transform = transforms.Compose([
    # transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor()])


test_transform_cifar10 = transforms.Compose([
    #transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_svhn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4376821, 0.4437697, 0.47280442], [0.19803012, 0.20101562, 0.19703614])])

test_transform_gtsrb = transforms.Compose([
    #transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.34025789, 0.31214415, 0.32143838], [0.2723574, 0.26082083, 0.2669115])])

test_transform_stl10 = transforms.Compose([
    #transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

empty_transform = transforms.Compose([
    #transforms.Resize(32),
    transforms.ToTensor()])

test_transform_imagenet = transforms.Compose([
    transforms.ToTensor(),])

test_transform_CLIP = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])

test_transform_food101 = transforms.Compose([
    #transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.54396613, 0.44242467, 0.34232726], [0.25573811, 0.25711934, 0.26267651])])

def get_surrogate_dataset(args):
    """
    obtain the surrogate dataset used to train the surrogate encoder
    :type base_dataset: String
    :type base_model: String
    :type base_epoch: Int
    :type surrogate_dataset: String
    :type k: Int
    :rtype: SurrogateDataset
    """
    return SurrogateDataset(img_dir=args.img_dir, victim_feature_bank_dir=args.victim_feature_bank_dir, transform=empty_transform, view_transform=finetune_transform, k=args.k)


def get_custom_dataset(data_dir, transform_type):
    print(f'Transform type: {transform_type}')

    if transform_type == 'cifar10':
        test_transform = test_transform_cifar10

    elif transform_type == 'svhn':
        test_transform = test_transform_svhn

    elif transform_type == 'stl10':
        test_transform = test_transform_stl10

    elif transform_type == 'gtsrb':
        test_transform = test_transform_gtsrb

    elif transform_type == 'food101':
        test_transform = test_transform_food101

    elif transform_type == 'empty' or transform_type is None:
        test_transform = empty_transform

    else:
        raise NotImplementedError

    data = CIFAR10Mem(numpy_file=data_dir, class_type=None, transform=test_transform)

    return data
