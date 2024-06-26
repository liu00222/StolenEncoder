import torch
import torchvision

from .cifar10_dataset_backdoor import get_clean_cifar10, get_backdoor_cifar10,get_backdoor_cifar10_ftt, get_backdoor_cifar10_evaluation,get_multi_backdoor_cifar10
from .gtsrb_dataset_backdoor import get_clean_gtsrb, get_backdoor_gtsrb, get_backdoor_gtsrb_ftt,get_backdoor_gtsrb_evaluation
from .svhn_dataset_backdoor import get_clean_svhn, get_backdoor_svhn, get_backdoor_svhn_evaluation, get_backdoor_svhn_ftt
from .stl10_dataset_backdoor import get_clean_stl10, get_backdoor_stl10, get_backdoor_stl10_evaluation, get_backdoor_stl10_ftt
from .surrogate_dataset import get_surrogate_dataset, get_custom_dataset


def get_dataset_clean(args):
    if args.dataset == 'cifar10':
        return get_clean_cifar10()
    elif args.dataset == 'gtsrb':
        return get_clean_gtsrb()
    elif args.dataset == 'svhn':
        return get_clean_svhn()
    elif args.dataset == 'stl10':
        return get_clean_stl10()
    else:
        raise NotImplementedError


def get_dataset_backdoor(args):
    if args.ftdataset =='cifar10':
        return get_backdoor_cifar10_ftt( args)
    elif args.ftdataset == 'gtsrb':
        return get_backdoor_gtsrb_ftt( args)
    elif args.ftdataset == 'svhn':
        return get_backdoor_svhn_ftt(args)
    elif args.ftdataset == 'stl10':
        return get_backdoor_stl10_ftt(args)
    else:
        raise NotImplementedError


def get_dataset_evaluation(args):
    if args.downstream_dataset =='cifar10':
        return get_backdoor_cifar10_evaluation( args)
    elif args.downstream_dataset == 'gtsrb':
        return get_backdoor_gtsrb_evaluation( args)
    elif args.downstream_dataset == 'svhn':
        return get_backdoor_svhn_evaluation(args)
    elif args.downstream_dataset == 'stl10':
        return get_backdoor_stl10_evaluation(args)
    else:
        raise NotImplementedError


def get_dataset_multi_backdoor(args):
    if args.ftdataset =='cifar10':
        return get_multi_backdoor_cifar10( args)
    elif args.ftdataset == 'gtsrb':
        return get_backdoor_gtsrb( args)
    else:
        raise NotImplementedError
