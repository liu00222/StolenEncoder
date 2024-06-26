


from .cifar10_clean_model import SimCLR
from .surrogate_model import SimCLRSurrogate
# from .surrogate_midel import
# from .cifar10_backdoor_model import SimCLRBackdoor
#
# from .resnet_pretrained_model import SimCLRPT
#
# from .cifar10_backdoor_model_loss3 import SimCLRBackdoorLoss3

def get_model_clean(args):
    return SimCLR(args.feature_dim, args.arch)

def get_surrogate_model(args):
    return SimCLRSurrogate(args.feature_dim, args.arch)


# def get_model_backdoor(args):
#
#     if args.ftdataset == 'cifar10':
#         return SimCLRBackdoor(args.feature_dim, args.arch)
#     elif args.ftdataset == 'gtsrb':
#         return SimCLRBackdoor(args.feature_dim, args.arch)
#     elif args.ftdataset == 'pubfig':
#         return SimCLRBackdoor(args.feature_dim, args.arch)
#     elif args.ftdataset == 'svhn':
#         return SimCLRBackdoor(args.feature_dim, args.arch)
#     elif args.ftdataset == 'stl10':
#         return SimCLRBackdoor(args.feature_dim, args.arch)
#     elif args.ftdataset == 'mask2':
#         return SimCLRBackdoor(args.feature_dim, args.arch)
#     else:
#         raise NotImplementedError
#
#
# def get_model_backdoor_loss3(args):
#
#     if args.ftdataset == 'cifar10':
#         return SimCLRBackdoorLoss3(args.feature_dim, args.arch)
#     elif args.ftdataset == 'gtsrb':
#         return SimCLRBackdoorLoss3(args.feature_dim, args.arch)
#     elif args.ftdataset == 'pubfig':
#         return SimCLRBackdoorLoss3(args.feature_dim, args.arch)
#     elif args.ftdataset == 'svhn':
#         return SimCLRBackdoorLoss3(args.feature_dim, args.arch)
#     elif args.ftdataset == 'stl10':
#         return SimCLRBackdoorLoss3(args.feature_dim, args.arch)
#     elif args.ftdataset == 'mask4072':
#         return SimCLRBackdoorLoss3(args.feature_dim, args.arch)
#     else:
#         raise NotImplementedError
#
# from .cifar10_multi_backdoor_model import SimCLRMultiBackdoor
# def get_model_backdoor_loss3_multiple(args):
#     if args.ftdataset == 'cifar10':
#         return SimCLRMultiBackdoor(args.feature_dim, args.arch)
#     elif args.ftdataset == 'gtsrb':
#         return SimCLRMultiBackdoor(args.feature_dim, args.arch)
#     elif args.ftdataset == 'pubfig':
#         return SimCLRMultiBackdoor(args.feature_dim, args.arch)
#     elif args.ftdataset == 'svhn':
#         return SimCLRMultiBackdoor(args.feature_dim, args.arch)
#     elif args.ftdataset == 'stl10':
#         return SimCLRMultiBackdoor(args.feature_dim, args.arch)
#     elif args.ftdataset == 'mask4072':
#         return SimCLRMultiBackdoor(args.feature_dim, args.arch)
#     else:
#         raise NotImplementedError
