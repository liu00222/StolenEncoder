import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.models.resnet import resnet50
from torchvision.models.resnet import resnet18, resnet34, resnet50
# from torchvision.models import vgg16, vgg19_bn, vgg19, vgg11, vgg11_bn, vgg16_bn
from torchvision.models import squeezenet1_0, googlenet, mnasnet1_0, inception_v3, densenet121, densenet201, alexnet
from torchvision.models import mobilenet_v2, shufflenet_v2_x1_0

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))

class SimCLRBase(nn.Module):

    def __init__(self, arch='resnet18'):
        super(SimCLRBase, self).__init__()

        self.f = []

        if 'resnet' in arch:
            print(f'using arch: {arch}')
            if arch == 'resnet18':
                model_name = resnet18()
            elif arch == 'resnet34':
                model_name = resnet34()
            elif arch == 'resnet50':
                model_name = resnet50()
            else:
                raise NotImplementedError
            for name, module in model_name.named_children():
                if name == 'conv1':
                    module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                    self.f.append(module)
        else:

            if arch == 'densenet121':
                print('using densenet121')
                model_name = densenet121()
                for name, module in model_name.named_children():
                    self.f.append(module)
                self.f.pop()
                self.f.append(nn.Flatten(start_dim=1))
                self.f.append(nn.Linear(1024, 512, bias=True))

            elif arch == 'mobilenet_v2':
                print('using mobilenet_v2')
                model_name = mobilenet_v2()
                for name, module in model_name.named_children():
                    self.f.append(module)
                tmp = self.f.pop()
                self.f.append(nn.Flatten(start_dim=1))
                self.f.append(tmp)
                self.f.append(nn.Linear(1000, 512, bias=True))

            elif arch == 'vgg19':
                print('using vgg19')
                model_name = vgg19()
                # self.f.append(nn.Flatten(start_dim=1))
                for name, module in model_name.named_children():
                    if name == 'classifier':
                        continue
                    self.f.append(module)

            elif arch == 'vgg19_bn':
                print('using vgg19_bn')
                model_name = vgg19_bn()
                # self.f.append(nn.Flatten(start_dim=1))
                for name, module in model_name.named_children():
                    if name == 'classifier':
                        continue
                    self.f.append(module)

            elif arch == 'alexnet':
                print('using alexnet')
                model_name = alexnet()
                for name, module in model_name.named_children():
                    self.f.append(module)
                # self.f.pop()
                # self.f.pop()
                self.f.append(nn.Flatten(start_dim=1))
                self.f.append(nn.Linear(1000, 512, bias=True))

            elif arch == 'mnasnet1_0':
                print('using mnasnet1_0')
                model_name = mnasnet1_0()
                for name, module in model_name.named_children():
                    self.f.append(module)
                self.f.append(nn.Flatten(start_dim=1))
                self.f.append(nn.Linear(1000, 512, bias=True))

            elif arch == 'shufflenet_v2_x1_0':
                print('using shufflenet_v2_x1_0')
                model_name = shufflenet_v2_x1_0()
                for name, module in model_name.named_children():
                    self.f.append(module)
                self.f.pop()
                self.f.append(nn.Flatten(start_dim=1))
                self.f.append(nn.Linear(1024, 512, bias=True))

        self.f = nn.Sequential(*self.f)
        print(self.f)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)

        return feature

class SimCLR(nn.Module):
    def __init__(self, feature_dim=128, arch='resnet18'):
        super(SimCLR, self).__init__()

        self.f = SimCLRBase(arch)
        # self.f = resnet18(pretrained=False)
        # conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.f.conv1 = conv1
        # if arch == 'resnet18':
        #     projection_model = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        # elif arch == 'resnet34':
        if arch == 'resnet50':
            projection_model = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        else:
            projection_model = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

        self.g = projection_model
        # encoder
        # projection head
        # self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        # g for ResNet-50.
        #self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):

        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


        # x_1, x_2 = self.f(x_1), self.f(x_2)
        # feature_1, feature_2 = torch.flatten(x_1, start_dim=1), torch.flatten(x_2, start_dim=1)
        # out_1, out_2 = self.g(feature_1), self.g(feature_2)

        # feature_1, out_1 = F.normalize(feature_1, dim=-1), F.normalize(out_1, dim=-1)
        # feature_2, out_2 = F.normalize(feature_2, dim=-1), F.normalize(out_2, dim=-1)
        # #feature_1, out_1 = net(pos_1)
        # #feature_2, out_2 = net(pos_2)
        # # [2*B, D]
        # out = torch.cat([out_1, out_2], dim=0)
        # # [2*B, 2*B]
        # sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.knn_t)
        # mask = (torch.ones_like(sim_matrix) - torch.eye(2 * args.batch_size, device=sim_matrix.device)).bool()
        # # [2*B, 2*B-1]
        # sim_matrix = sim_matrix.masked_select(mask).view(2 * args.batch_size, -1)

        # # compute loss
        # pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.knn_t)
        # # [2*B]
        # pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        # loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        # return loss
        # #return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
