from torchvision import transforms
from .backdoor_dataset import CIFAR10Mem, CIFAR10Pair, BadEncoderTestBackdoor, ReferenceImg
import numpy as np

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4376821, 0.4437697, 0.47280442], [0.19803012, 0.20101562, 0.19703614])])

finetune_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4376821, 0.4437697, 0.47280442], [0.19803012, 0.20101562, 0.19703614])])

test_transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_stl10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

backdoor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4376821, 0.4437697, 0.47280442], [0.19803012, 0.20101562, 0.19703614])])

test_transform_imagenet = transforms.Compose([
    transforms.ToTensor(),])

test_transform_CLIP = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# def get_clean_svhn(data_dir):
#     train_data = CIFAR10Pair(numpy_file=data_dir + "train.npz", class_type= classes, transform=train_transform)
#     memory_data = CIFAR10Mem(numpy_file=data_dir + "train.npz", class_type= classes, transform=test_transform_svhn)
#     test_data  = CIFAR10Mem(numpy_file=data_dir + "test.npz", class_type= classes,transform=test_transform_svhn)

#     return train_data, memory_data, test_data


def get_downstream_svhn(args):
    training_file_name = 'train.npz'
    testing_file_name = 'test.npz'

    if args.pretraining_dataset == 'cifar10':
        print('test_transform_cifar10')
        test_transform = test_transform_cifar10
    elif args.pretraining_dataset == 'stl10':
        print('test_transform_stl10')
        test_transform = test_transform_stl10
    elif args.pretraining_dataset == 'CLIP':
        print('test_transform_CLIP')
        test_transform = test_transform_CLIP
        training_file_name = 'train_224.npz'
        testing_file_name = 'test_224.npz'
    elif args.pretraining_dataset == 'imagenet':
        print('test_transform_imagenet')
        test_transform = test_transform_imagenet
        training_file_name = 'train_224.npz'
        testing_file_name = 'test_224.npz'
    else:
        raise NotImplementedError

    target_dataset = ReferenceImg(reference_file=args.reference_file, transform=test_transform)
    memory_data = CIFAR10Mem(numpy_file=args.data_dir+training_file_name, class_type=classes, transform=test_transform)
    test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir+testing_file_name, trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform)
    test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+testing_file_name, class_type=classes, transform=test_transform)

    return target_dataset, memory_data, test_data_clean, test_data_backdoor
