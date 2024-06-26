
from torchvision import transforms
from .backdoor_dataset import CIFAR10PairBD, CIFAR10PairBDFTT, CIFAR10Mem, CIFAR10Pair, CIFAR10TestBackdoor
from .backdoor_dataset import CIFAR10PairBDMulti
import numpy as np
from PIL import Image


imagenet_train_path = '/path/to/imagenet/train'
imagenet_val_path = '/path/to/imagenet/val'

image_resolution = 224

train_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

finetune_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    ])

backdoor_transform = transforms.Compose([
    transforms.ToTensor(),
    ])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    ])
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


from .backdoor_dataset import CIFAR10PairBDMultiImageNet

def get_multi_backdoor_imagenet(args):
    training_data_num = 1281167
    val_data_num = 50000
    np.random.seed(100)

    if "testing_dataset" in args.trial:
        print('loading from the testing data')
        training_data_sampling_indices = np.random.choice(val_data_num, int(np.floor(val_data_num * args.fraction / 100.0)), replace=False)
        train_data_clean = CIFAR10PairBDMultiImageNet(filepath = imagenet_val_path, \
            trigger_file=args.trigger_file, target_file= args.target_file, class_type=classes,indices = training_data_sampling_indices, transform=train_transform, bd_transform=backdoor_transform,ftt_transform=finetune_transform)

    else:
        print("loading from the training data")
        print('number of sampled training examples:')
        print(int(np.floor(training_data_num*args.fraction/100.0)))
        num_of_samples =  int(np.floor(training_data_num*args.fraction/100.0))
        training_data_sampling_indices = np.random.choice(training_data_num, num_of_samples, replace=False)
        train_data_clean = CIFAR10PairBDMultiImageNet(filepath = imagenet_train_path, \
            trigger_file=args.trigger_file, target_file= args.target_file, class_type=classes,indices = training_data_sampling_indices, transform=train_transform, bd_transform=backdoor_transform,ftt_transform=finetune_transform)
    return train_data_clean
