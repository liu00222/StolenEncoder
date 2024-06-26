from torchvision import transforms
from .backdoor_dataset import CIFAR10Mem, CIFAR10Pair, BadEncoderTestBackdoor, ReferenceImg
import numpy as np

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.34025789, 0.31214415, 0.32143838], [0.2723574, 0.26082083, 0.2669115])])

finetune_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.34025789, 0.31214415, 0.32143838], [0.2723574, 0.26082083, 0.2669115])])

test_transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_stl10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

backdoor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.34025789, 0.31214415, 0.32143838], [0.2723574, 0.26082083, 0.2669115])])

test_transform_imagenet = transforms.Compose([
    transforms.ToTensor(),])

test_transform_CLIP = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])

classes = ['Speed limit (20km/h)',
                        'Speed limit (30km/h)',
                        'Speed limit (50km/h)',
                        'Speed limit (60km/h)',
                        'Speed limit (70km/h)',
                        'Speed limit (80km/h)', #5
                        'End of speed limit (80km/h)',
                        'Speed limit (100km/h)',
                        'Speed limit (120km/h)',
                        'No passing',
                        'No passing for vehicles over 3.5 metric tons', #10
                        'Right-of-way at the next intersection',
                        'Priority road',
                        'Yield',
                        'Stop', #14
                        'No vehicles',  #15
                        'Vehicles over 3.5 metric tons prohibited',
                        'No entry',
                        'General caution',
                        'Dangerous curve to the left',
                        'Dangerous curve to the right', #20
                        'Double curve',
                        'Bumpy road',
                        'Slippery road',
                        'Road narrows on the right',
                        'Road work',    #25
                        'Traffic signals',
                        'Pedestrians',
                        'Children crossing',
                        'Bicycles crossing',
                        'Beware of ice/snow',   #30
                        'Wild animals crossing',
                        'End of all speed and passing limits',
                        'Turn right ahead',
                        'Turn left ahead',
                        'Ahead only',   #35
                        'Go straight or right',
                        'Go straight or left',
                        'Keep right',
                        'Keep left',
                        'Roundabout mandatory', #40
                        'End of no passing',
                        'End of no passing by vehicles over 3.5 metric tons']




# def get_clean_gtsrb(data_dir):
#     train_data = CIFAR10Pair(numpy_file=data_dir + "train.npz", class_type= classes, transform=train_transform)
#     memory_data = CIFAR10Mem(numpy_file=data_dir + "train.npz", class_type= classes, transform=test_transform_gtsrb)
#     test_data  = CIFAR10Mem(numpy_file=data_dir + "test.npz", class_type= classes,transform=test_transform_gtsrb)

#     return train_data, memory_data, test_data


def get_downstream_gtsrb(args):
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
