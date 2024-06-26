
from torchvision import transforms
from .backdoor_dataset import CIFAR10PairBD, CIFAR10PairBDFTT,CIFAR10Mem, CIFAR10Pair, CIFAR10TestBackdoor,CIFAR10TargetImage
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

backdoor_transform = transforms.Compose([
    #transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.34025789, 0.31214415, 0.32143838], [0.2723574, 0.26082083, 0.2669115])])

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




def get_clean_gtsrb():
    data_dir = '/path/to/gtsrb/'
    train_data = CIFAR10Pair(numpy_file=data_dir + "train.npz", class_type= classes, transform=train_transform)
    memory_data = CIFAR10Mem(numpy_file=data_dir + "train.npz", class_type= classes, transform=test_transform_gtsrb)
    test_data  = CIFAR10Mem(numpy_file=data_dir + "test.npz", class_type= classes,transform=test_transform_gtsrb)

    return train_data, memory_data, test_data

def get_backdoor_gtsrb(args):
    if args.basedataset == 'cifar10':
        print('test_transform_cifar10')
        test_transform = test_transform_cifar10
    elif args.basedataset == 'svhn':
        print('test_transform_svhn')
        test_transform = test_transform_svhn
    elif args.basedataset == 'stl10':
        print('test_transform_stl10')
        test_transform = test_transform_stl10
    elif args.basedataset == 'gtsrb':
        print('test_transform_gtsrb')
        test_transform = test_transform_gtsrb
    else:
        raise NotImplementedError

    data_dir = '/path/to/gtsrb/'
    training_data_num = 39209
    np.random.seed(100)
    training_data_sampling_indices = np.random.choice(training_data_num, int(np.floor(training_data_num*args.fraction/100.0)), replace=False)
    train_data_clean = CIFAR10PairBD(numpy_file=data_dir + "train.npz", \
        trigger_file=args.trigger_file, target_file= args.target_file, class_type=classes,indices = training_data_sampling_indices, transform=train_transform, bd_transform=backdoor_transform)
    memory_data = CIFAR10Mem(numpy_file=data_dir +"train.npz", class_type=classes, transform=test_transform)
    test_data_backdoor = CIFAR10TestBackdoor(numpy_file=data_dir+"/test.npz", trigger_file=args.trigger_file, target_file= args.target_file  , class_type=classes, transform=test_transform)
    test_data_clean = CIFAR10Mem(numpy_file=data_dir+"test.npz", class_type=classes, transform=test_transform)

    return train_data_clean, memory_data, test_data_clean, test_data_backdoor

def get_backdoor_gtsrb_ftt(args):
    if args.basedataset == 'cifar10':
        print('test_transform_cifar10')
        test_transform = test_transform_cifar10
    elif args.basedataset == 'svhn':
        print('test_transform_svhn')
        test_transform = test_transform_svhn
    elif args.basedataset == 'stl10':
        print('test_transform_stl10')
        test_transform = test_transform_stl10
    elif args.basedataset == 'gtsrb':
        print('test_transform_gtsrb')
        test_transform = test_transform_gtsrb
    else:
        raise NotImplementedError
    data_dir = '/path/to/gtsrb/'
    training_data_num = 39209
    np.random.seed(100)
    training_data_sampling_indices = np.random.choice(training_data_num, int(np.floor(training_data_num*args.fraction/100.0)), replace=False)
    train_data_clean = CIFAR10PairBDFTT(numpy_file=data_dir + "train.npz", \
        trigger_file=args.trigger_file, target_file= args.target_file, class_type=classes,indices = training_data_sampling_indices, transform=train_transform, bd_transform=backdoor_transform,ftt_transform=finetune_transform)
    memory_data = CIFAR10Mem(numpy_file=data_dir +"train.npz", class_type=classes, transform=test_transform)
    test_data_backdoor = CIFAR10TestBackdoor(numpy_file=data_dir+"/test.npz", trigger_file=args.trigger_file, target_file= args.target_file  , class_type=classes, transform=test_transform)
    test_data_clean = CIFAR10Mem(numpy_file=data_dir+"test.npz", class_type=classes, transform=test_transform)

    return train_data_clean, memory_data, test_data_clean, test_data_backdoor



def get_backdoor_gtsrb_evaluation(args):
    if args.basedataset == 'cifar10':
        print('test_transform_cifar10')
        test_transform = test_transform_cifar10
    elif args.basedataset == 'svhn':
        print('test_transform_svhn')
        test_transform = test_transform_svhn
    elif args.basedataset == 'stl10':
        print('test_transform_stl10')
        test_transform = test_transform_stl10
    elif args.basedataset == 'gtsrb':
        print('test_transform_gtsrb')
        test_transform = test_transform_gtsrb
    else:
        raise NotImplementedError
    _, memory_data, test_data_clean, test_data_backdoor = get_backdoor_gtsrb(args)

    target_dataset = CIFAR10TargetImage(target_file = args.target_file, transform = test_transform)

    return target_dataset, memory_data, test_data_clean, test_data_backdoor
