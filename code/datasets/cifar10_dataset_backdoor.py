
from torchvision import transforms
from .backdoor_dataset import CIFAR10PairBD, CIFAR10PairBDFTT, CIFAR10Mem, CIFAR10Pair, CIFAR10TestBackdoor, CIFAR10TargetImage
from .backdoor_dataset import CIFAR10PairBDMulti
import numpy as np

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

train_transform_downstrem_stl10 = transforms.Compose([
    transforms.RandomResizedCrop(96),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

finetune_transform = transforms.Compose([
    #transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

backdoor_transform = transforms.Compose([
    #transforms.RandomResizedCrop(32),
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    #transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

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

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

data_dir = '/path/to/cifar10/'

def get_clean_cifar10():

    train_data = CIFAR10Pair(numpy_file=data_dir + "train.npz", class_type= classes, transform=train_transform)
    memory_data = CIFAR10Mem(numpy_file=data_dir + "train.npz", class_type= classes, transform=test_transform_cifar10)
    test_data  = CIFAR10Mem(numpy_file=data_dir + "test.npz", class_type= classes,transform=test_transform_cifar10)

    return train_data, memory_data, test_data

def get_backdoor_cifar10(args):
    if args.base_dataset == 'cifar10':
        print('test_transform_cifar10')
        test_transform = test_transform_cifar10
    elif args.base_dataset == 'svhn':
        print('test_transform_svhn')
        test_transform = test_transform_svhn
    elif args.base_dataset == 'stl10':
        print('test_transform_stl10')
        test_transform = test_transform_stl10
    elif args.base_dataset == 'gtsrb':
        print('test_transform_gtsrb')
        test_transform = test_transform_gtsrb
    else:
        raise NotImplementedError
    #data_dir = '/path/to/cifar10/'
    training_data_num = 50000
    np.random.seed(100)
    print('number of training examples:')
    print(int(np.floor(training_data_num*args.fraction/100.0)))
    training_data_sampling_indices = np.random.choice(training_data_num, int(np.floor(training_data_num*args.fraction/100.0)), replace=False)
    train_data_clean = CIFAR10PairBD(numpy_file=data_dir + "train.npz", \
        trigger_file=args.trigger_file, target_file= args.target_file, class_type=classes,indices = training_data_sampling_indices, transform=train_transform, bd_transform=backdoor_transform)
    memory_data = CIFAR10Mem(numpy_file=data_dir +"train.npz", class_type=classes, transform=test_transform)
    test_data_backdoor = CIFAR10TestBackdoor(numpy_file=data_dir+"/test.npz", trigger_file=args.trigger_file, target_file= args.target_file  , class_type=classes, transform=test_transform)
    test_data_clean = CIFAR10Mem(numpy_file=data_dir+"test.npz", class_type=classes, transform=test_transform)

    return train_data_clean, memory_data, test_data_clean, test_data_backdoor


def get_backdoor_cifar10_ftt(args):
    if args.base_dataset == 'cifar10':
        print('test_transform_cifar10')
        test_transform = test_transform_cifar10
    elif args.base_dataset == 'svhn':
        print('test_transform_svhn')
        test_transform = test_transform_svhn
    elif args.base_dataset == 'stl10':
        print('test_transform_stl10')
        test_transform = test_transform_stl10
    elif args.base_dataset == 'gtsrb':
        print('test_transform_gtsrb')
        test_transform = test_transform_gtsrb
    else:
        raise NotImplementedError
    #data_dir = '/path/to/cifar10/'
    training_data_num = 50000
    testing_data_num = 10000
    np.random.seed(100)
    print('number of training examples:')
    print(int(np.floor(training_data_num*args.fraction/100.0)))
    training_data_sampling_indices = np.random.choice(training_data_num,
                                                      int(np.floor(training_data_num * args.fraction / 100.0)),
                                                      replace=False)

    if "testing_dataset" in args.trial:
        print('loading from the testing data')
        training_data_sampling_indices = np.random.choice(testing_data_num, int(np.floor(testing_data_num * args.fraction / 100.0)), replace=False)
        train_data_clean = CIFAR10PairBDFTT(numpy_file=data_dir + "test.npz", \
                                            trigger_file=args.trigger_file, target_file=args.target_file,
                                            class_type=classes, indices=training_data_sampling_indices,
                                            transform=train_transform, bd_transform=backdoor_transform,
                                            ftt_transform=finetune_transform)
    elif "food101" in args.trial:
        print('loading from Food-101 dataset')
        food101_data_dir = '/path/to/food101/'
        food101_data_num = 10000
        training_data_sampling_indices = np.random.choice(food101_data_num, int(np.floor(testing_data_num * args.fraction / 100.0)), replace=False)

        food101_class  = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad',
           'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad',
           'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry',
           'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
           'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts',
           'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips',
           'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
           'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon',
           'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus',
           'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons',
           'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes',
           'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich',
           'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad',
           'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak',
           'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']

        train_data_clean = CIFAR10PairBDFTT(numpy_file=food101_data_dir + "train.npz", \
                                            trigger_file=args.trigger_file, target_file=args.target_file,
                                            class_type=food101_class, indices=training_data_sampling_indices,
                                            transform=train_transform, bd_transform=backdoor_transform,
                                            ftt_transform=finetune_transform)
    else:
        print('loading from the training data')
        train_data_clean = CIFAR10PairBDFTT(numpy_file=data_dir + "train.npz", \
            trigger_file=args.trigger_file, target_file= args.target_file, class_type=classes,indices = training_data_sampling_indices, transform=train_transform, bd_transform=backdoor_transform,ftt_transform=finetune_transform)

    memory_data = CIFAR10Mem(numpy_file=data_dir +"train.npz", class_type=classes, transform=test_transform)
    test_data_backdoor = CIFAR10TestBackdoor(numpy_file=data_dir+"/test.npz", trigger_file=args.trigger_file, target_file= args.target_file  , class_type=classes, transform=test_transform)
    test_data_clean = CIFAR10Mem(numpy_file=data_dir+"test.npz", class_type=classes, transform=test_transform)

    return train_data_clean, memory_data, test_data_clean, test_data_backdoor


def get_backdoor_cifar10_evaluation(args):
    if args.base_dataset == 'cifar10':
        print('test_transform_cifar10')
        test_transform = test_transform_cifar10
    elif args.base_dataset == 'svhn':
        print('test_transform_svhn')
        test_transform = test_transform_svhn
    elif args.base_dataset == 'stl10':
        print('test_transform_stl10')
        test_transform = test_transform_stl10
    elif args.base_dataset == 'gtsrb':
        print('test_transform_gtsrb')
        test_transform = test_transform_gtsrb
    else:
        raise NotImplementedError
    memory_data = CIFAR10Mem(numpy_file=data_dir +"train.npz", class_type=classes, transform=test_transform)
    test_data_clean = CIFAR10Mem(numpy_file=data_dir+"test.npz", class_type=classes, transform=test_transform)

    return memory_data, test_data_clean

def get_multi_backdoor_cifar10(args):
    if args.base_dataset == 'cifar10':
        print('test_transform_cifar10')
        test_transform = test_transform_cifar10
    elif args.base_dataset == 'svhn':
        print('test_transform_svhn')
        test_transform = test_transform_svhn
    elif args.base_dataset == 'stl10':
        print('test_transform_stl10')
        test_transform = test_transform_stl10
    elif args.base_dataset == 'gtsrb':
        print('test_transform_gtsrb')
        test_transform = test_transform_gtsrb
    else:
        raise NotImplementedError
    #data_dir = '/path/to/cifar10/'
    training_data_num = 50000
    np.random.seed(100)
    print('number of training examples:')
    print(int(np.floor(training_data_num*args.fraction/100.0)))
    training_data_sampling_indices = np.random.choice(training_data_num, int(np.floor(training_data_num*args.fraction/100.0)), replace=False)
    train_data_clean = CIFAR10PairBDMulti(numpy_file=data_dir + "train.npz", \
        trigger_file=args.trigger_file, target_file= args.target_file, class_type=classes,indices = training_data_sampling_indices, transform=train_transform, bd_transform=backdoor_transform,ftt_transform=finetune_transform)
    memory_data = CIFAR10Mem(numpy_file=data_dir +"train.npz", class_type=classes, transform=test_transform)
    test_data_backdoor = CIFAR10TestBackdoor(numpy_file=data_dir+"/test.npz", trigger_file=args.trigger_file, target_file= args.target_file  , class_type=classes, transform=test_transform)
    test_data_clean = CIFAR10Mem(numpy_file=data_dir+"test.npz", class_type=classes, transform=test_transform)

    return train_data_clean, memory_data, test_data_clean, test_data_backdoor
