import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101

from collections import OrderedDict
from typing import Tuple, Union



class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, enable_downsample=False):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        # self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        # print(Bottleneck.expansion)
        # print(self.expansion)

        if stride > 1 or inplanes != planes * Bottleneck.expansion or enable_downsample:
        # if stride > 1 or inplanes != planes * 4:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.avgpool(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        print(input_resolution // 32, embed_dim, heads, output_dim)
        self.attnpool = AttentionPool2d(input_resolution // 32, 512, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        # layers = [Bottleneck(self._inplanes, planes, stride, enable_downsample=True)]
        #
        # self._inplanes = planes * Bottleneck.expansion
        # for _ in range(1, blocks):
        #     layers.append(Bottleneck(self._inplanes, planes))
        # return nn.Sequential(*layers)
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class CLIPSurrogate(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 ):
        super().__init__()

        print('Modified ResNet34')
        vision_heads = vision_width * 32 // 64
        self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width)
        print(self.visual)
        # exit()

    def forward(self, img_raw, views, victim_feature, lambda_value, exp=False, k=-1):
        if not exp:
            raise NotImplementedError
        if exp:
            assert (k >= 0)
            surrogate_feature = self.visual(img_raw)
            surrogate_feature = F.normalize(surrogate_feature, dim=-1)

            for i in range(len(views)):
                views[i] = F.normalize(self.visual(views[i]), dim=-1)

            loss1 = loss2 = 0

            for i in range(len(surrogate_feature)):
                loss1 += torch.dist(surrogate_feature[i], victim_feature[i], 2)
                if k != 0:
                    loss2 += sum([torch.dist(f[i], victim_feature[i], 2) for f in views])
            if k == 0:
                return loss1
            else:
                return loss1 + ((lambda_value*loss2)/k)


class SimCLRBase(nn.Module):

    def __init__(self, arch='resnet18'):
        super(SimCLRBase, self).__init__()

        self.f = []

        if arch == 'resnet18':
            model_name = resnet18()
        elif 'resnet34' in arch:
            model_name = resnet34()
        elif arch == 'resnet50':
            model_name = resnet50()
        elif arch == 'resnet101':
            model_name = resnet101()
        else:
            raise NotImplementedError
        for name, module in model_name.named_children():
            # if name == 'conv1':
            #     module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d) and not isinstance(module, nn.AdaptiveAvgPool2d):
                self.f.append(module)
                # print('\n-----------')
                # print(type(module))
                # print(module)
                # print('-----------')
                # print()
            # if isinstance(module, nn.AdaptiveAvgPool2d):
            #     self.f.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        # self.f.append(
        #     nn.Sequential(
        #         nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False),
        #         nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
        #         nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #         nn.Sequential(
        #             nn.Conv2d(512, 1024, kernel_size=1, stride=2, bias=False),
        #             nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #         ),
        #         nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
        #         nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
        #         nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #         nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
        #         nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
        #         nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     )
        # )
        if arch == 'resnet101':
            self.f.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
            self.f.append(nn.Flatten(start_dim=1))
            self.f.append(nn.Linear(2048, 1024, bias=False))
        elif arch == 'resnet34_linear2':
            print(arch)
            self.f.append(nn.AdaptiveAvgPool2d(output_size=(2, 2)))
            self.f.append(nn.Flatten(start_dim=1))
            self.f.append(nn.Linear(2048, 1024, bias=False))
        elif arch == 'resnet34_linear':
            print(arch)
            self.f.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
            self.f.append(nn.Flatten(start_dim=1))
            self.f.append(nn.Linear(512, 1024, bias=False))
        elif arch == 'resnet34first2':
            self.f.append(nn.AdaptiveAvgPool2d(output_size=(2, 1)))
        else:
            self.f.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        # self.f.append(nn.AdaptiveAvgPool2d(output_size=(2, 2)))
        # self.f.append(nn.Flatten(start_dim=1))
        # self.f.append(nn.Linear(2048, 1024, bias=False))

        self.f = nn.Sequential(*self.f)
        # print()
        print(self.f)
        # exit()

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)

        return feature

class SimCLRSurrogate(nn.Module):
    def __init__(self, feature_dim=1024, arch='resnet18'):
        super(SimCLRSurrogate, self).__init__()


        self.visual = SimCLRBase(arch)
        # self.visual = resnet18(pretrained=False)
        # conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.visual.conv1 = conv1
        if arch == 'resnet18':
            projection_model = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
            self.g = projection_model
        elif 'resnet34' in arch:
            projection_model = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
            self.g = projection_model
        elif arch == 'resnet50':
            projection_model = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
            self.g = projection_model


        # encoder
        # projection head
        # self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        # g for ResNet-50.
        #self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))



    def forward(self, img_raw, views, victim_feature, lambda_value, exp=False, k=-1):
        if not exp:
            raise NotImplementedError
        if exp:
            # import numpy as np
            # print(victim_feature.shape)
            # print(torch.max(victim_feature))
            # print(torch.min(victim_feature))
            # exit()
            assert (k >= 0)
            surrogate_feature = self.visual(img_raw)
            surrogate_feature = F.normalize(surrogate_feature, dim=-1)

            for i in range(len(views)):
                views[i] = F.normalize(self.visual(views[i]), dim=-1)

            loss1 = loss2 = 0

            # print(victim_feature[0].shape)
            # print(surrogate_feature[0].shape)

            for i in range(len(surrogate_feature)):
                loss1 += torch.dist(surrogate_feature[i], victim_feature[i], 2)
                if k != 0:
                    loss2 += sum([torch.dist(f[i], victim_feature[i], 2) for f in views])
            if k == 0:
                return loss1
            else:
                return loss1 + ((lambda_value*loss2)/k)
