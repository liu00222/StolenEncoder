import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.models.resnet import resnet50
from torchvision.models.resnet import resnet18, resnet34, resnet50

class SimCLRBase(nn.Module):

    def __init__(self, arch='resnet18'):
        super(SimCLRBase, self).__init__()

        self.f = []

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
        self.f = nn.Sequential(*self.f)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)

        return feature        

class SimCLRBackdoor(nn.Module):
    def __init__(self, feature_dim=128, arch='resnet18'):
        super(SimCLRBackdoor, self).__init__()


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




    def forward(self, im_1, img_raw, img_backdoor, img_target,img_target_1, clean_net, args):

        with torch.no_grad():
            # clean_feature_1 =  clean_net.f(im_1)
            # clean_feature_1 = F.normalize(clean_feature_1, dim=-1)

            clean_feature_raw = clean_net.f(img_raw)
            clean_feature_raw = F.normalize(clean_feature_raw, dim=-1)

            clean_feature_target = clean_net.f(img_target)
            clean_feature_target = F.normalize(clean_feature_target, dim=-1)
      
        # feature_1 = self.f(im_1)
        # feature_1 = F.normalize(feature_1, dim=-1)

        feature_raw = self.f(img_raw)
        feature_raw = F.normalize(feature_raw, dim=-1)


        feature_backdoor = self.f(img_backdoor)
        feature_backdoor = F.normalize(feature_backdoor, dim=-1)

        feature_target = self.f(img_target)
        feature_target = F.normalize(feature_target, dim=-1)

        feature_target_1 = self.f(img_target_1)
        feature_target_1 = F.normalize(feature_target_1, dim=-1)

        loss_1 = - torch.sum(feature_backdoor * feature_target_1, dim=-1).mean()
        loss_2 = - torch.sum(feature_target * clean_feature_target, dim=-1).mean()
        loss_3 = - torch.sum(feature_target_1 * clean_feature_target, dim=-1).mean()
        loss_4 = - torch.sum(feature_raw * clean_feature_raw, dim=-1).mean()
        #loss_5 = - torch.sum(feature_1 * clean_feature_1, dim=-1).mean()


        #loss = loss_1 + loss_2 + loss_4 + args.lambdav *  loss_3 + loss_5
        loss = loss_1 + loss_2 + loss_3 + loss_4

        return loss , loss_1, loss_2, loss_3, loss_4
        
