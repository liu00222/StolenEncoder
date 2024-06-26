from .resnet_model import SimCLR
from .clip_model import CLIP
from .imagenet_model import ImageNetResNet
from .surrogate_model import SimCLRSurrogate, CLIPSurrogate
# from .backdoor_model import SimCLRBackdoor
#from .multi_backdoor_model import SimCLRBadEncoder



# def get_model_clean(args):
#     return SimCLR()

def get_encoder_architecture(args):
    if args.pretraining_dataset == 'cifar10':
        return SimCLR()
    elif args.pretraining_dataset == 'stl10':
        return SimCLR()
    elif args.pretraining_dataset == 'imagenet':
        return ImageNetResNet(1024, 224, vision_layers=(3, 4, 6, 3), vision_width=64)
    elif args.pretraining_dataset == 'CLIP':
        return CLIP(1024, 224, vision_layers=(3, 4, 6, 3), vision_width=64)
    else:
        raise ValueError('Unknown pretraining dataset: {}'.format(args.pretraining_dataset))

def get_CLIP_surrogate(args):
    if 'resnet34' in args.arch:
        print('using the resnet34 model')
        return SimCLRSurrogate(arch=args.arch)
    elif 'resnet18' in args.arch:
        print('using the resnet18 model')
        return SimCLRSurrogate(arch='resnet18')
    elif 'resnet50' in args.arch:
        print('using the resnet50 model')
        return SimCLRSurrogate(arch='resnet50')
    elif 'resnet101' in args.arch:
        print('using the resnet101 model')
        return SimCLRSurrogate(arch='resnet101')
    elif 'CLIP' in args.arch:
        print('using the CLIP model')
        return CLIPSurrogate(1024, 224, vision_layers=(3, 4, 6, 3), vision_width=64)
        # return CLIP(1024, 224, vision_layers=(3, 4, 6, 3), vision_width=64)
    else:
        raise NotImplementedError

# def get_model_backdoor(args):
#     return SimCLRBackdoor()

#def get_model_backdoor(args):
    #return SimCLRMultiBackdoor()
    #return SimCLRBadEncoder()

def get_model_pretrained_CLIP(args):
    return CLIP(1024, 224, vision_layers=(3, 4, 6, 3), vision_width=64)


def get_model_pretrained_imagenet(args):
    return ImageNetResNet(1024, 224, vision_layers=(3, 4, 6, 3), vision_width=64)
