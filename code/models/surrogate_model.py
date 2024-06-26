import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, vgg19_bn, vgg19, vgg11, vgg16_bn
#from torchvision.models.resnet import resnet50
from torchvision.models.resnet import resnet18, resnet34, resnet50

class SimCLRBase(nn.Module):

    def __init__(self, arch='resnet18'):
        super(SimCLRBase, self).__init__()

        self.f = []

        if 'resnet' in arch:
            if arch == 'resnet18':
                print('using resnet18')
                model_name = resnet18()
            elif arch == 'resnet34':
                print('using resnet34')
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
        elif 'vgg' in arch:
            model_name = vgg19_bn()
            # self.f.append(nn.Flatten(start_dim=1))
            for name, module in model_name.named_children():
                if name == 'classifier':
                    continue
                # if name == 'conv1':
                #     module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                self.f.append(module)
                # if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                #     self.f.append(module)
            # self.f.pop()
            # self.f.pop()
            # tmp = self.f.pop()
            # self.f.append(tmp)
            # self.f.append(nn.Linear(1000, 512, bias=True))
        self.f = nn.Sequential(*self.f)
        # print(self.f)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)

        return feature

class SimCLRSurrogate(nn.Module):
    def __init__(self, feature_dim=128, arch='resnet18'):
        super(SimCLRSurrogate, self).__init__()


        self.f = SimCLRBase(arch)
        # self.f = resnet18(pretrained=False)
        # conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.f.conv1 = conv1
        if arch == 'resnet18':
            projection_model = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        elif arch == 'resnet34':
            projection_model = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        elif arch == 'resnet50':
            projection_model = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        else:
            raise NotImplementedError

        self.g = projection_model
        # encoder
        # projection head
        # self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        # g for ResNet-50.
        #self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))




    def forward(self, img_raw, views, victim_feature, lambda_value, k, distance='l2'):
        surrogate_feature = self.f(img_raw)
        surrogate_feature = F.normalize(surrogate_feature, dim=-1)

        for i in range(len(views)):
            views[i] = F.normalize(self.f(views[i]), dim=-1)

        if distance == 'cosine':
            loss1 = - torch.sum(surrogate_feature * victim_feature, dim=-1).mean()
            loss2 = 0
            for i in range(len(views)):
                loss2 -= torch.sum(views[i] * victim_feature, dim=-1).mean()
        else:
            loss1 = loss2 = 0
            for i in range(len(surrogate_feature)):
                if distance == 'l2':
                    loss1 += torch.dist(surrogate_feature[i], victim_feature[i], 2)
                    if k != 0:
                        loss2 += sum([torch.dist(f[i], victim_feature[i], 2) for f in views])

                elif distance == 'l1':
                    loss1 += torch.dist(surrogate_feature[i], victim_feature[i], 1)
                    if k != 0:
                        loss2 += sum([torch.dist(f[i], victim_feature[i], 1) for f in views])

                else:
                    raise NotImplementedError

        if k != 0:
            return loss1 + ((lambda_value*loss2)/k)
        else:
            return loss1
