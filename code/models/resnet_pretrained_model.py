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
            model_name = resnet18(pretrained=True)
        elif arch == 'resnet34':
            model_name = resnet34(pretrained=True)
        elif arch == 'resnet50':
            model_name = resnet50(pretrained=True)
        else:
            raise NotImplementedError
        for name, module in model_name.named_children():
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        self.f = nn.Sequential(*self.f)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)

        return feature  

class SimCLRPT(nn.Module):
    def __init__(self, feature_dim=128, arch='resnet18'):
        super(SimCLRPT, self).__init__()

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
