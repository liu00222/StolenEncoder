import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


from .trigger_utility import add_backdoor_trigger_to_data

import torch 

from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F

def add_backdoor_trigger(data, trigger, pp):
    return add_backdoor_trigger_to_data(data, trigger, pp)


def visualize_tnse(feature, targets, filename):
    ret = TSNE(n_components=2, random_state=0).fit_transform(feature)
    target_ids = range(len(set(targets)))
    
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'violet', 'orange', 'purple']
    
    plt.figure(figsize=(12, 10))
    
    ax = plt.subplot(aspect='equal')
    for label in set(targets):
        if label >= 10:
            break
        idx = np.where(np.array(targets) == label)[0]
        plt.scatter(ret[idx, 0], ret[idx, 1], c=colors[label], label=label)
    
    # for i in range(0, len(targets), 250):
    #     img = (mnist[i][0] * 0.3081 + 0.1307).numpy()[0]
    #     img = OffsetImage(img, cmap=plt.cm.gray_r, zoom=0.5) 
    #     ax.add_artist(AnnotationBbox(img, ret[i]))
    plt.xlim([-100, 100]) 
    plt.ylim([-100, 100]) 
    plt.legend()
    #plt.show()
    plt.savefig(filename)
    plt.close()

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    # if args.cos:  # cosine lr schedule
    #     lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    # else:  # stepwise lr schedule
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

# test using a knn monitor
def test(net, memory_data_loader, test_data_clean_loader, test_data_backdoor_loader, epoch, args):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_clean_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))

        total_num, total_top1 = 0., 0.
        test_bar = tqdm(test_data_backdoor_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))

    return total_top1 / total_num * 100


def train_multiple(net, clean_net, data_loader, train_optimizer, epoch, args):
    net.g.eval()
    net.f.train()


    for module in net.f.modules():
    # print(module)
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()


    clean_net.eval()

    adjust_learning_rate(train_optimizer, epoch, args)

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    total_loss_1, total_loss_2, total_loss_3, total_loss_4 = 0.0, 0.0, 0.0, 0.0
    #target_img = target_img.cuda(non_blocking=True)
    for img_raw, img_backdoor_list, target_image_list,target_img_1_list in train_bar:
        img_raw = img_raw.cuda(non_blocking=True)
        target_image_cuda_list, target_img_1_cuda_list, img_backdoor_cuda_list = [], [], []
        for target_image in target_image_list:
            target_image_cuda_list.append(target_image.cuda(non_blocking=True))
        for target_img_1 in target_img_1_list:
            target_img_1_cuda_list.append(target_img_1.cuda(non_blocking=True))
        for img_backdoor in img_backdoor_list:
            img_backdoor_cuda_list.append(img_backdoor.cuda(non_blocking=True))
        #loss, loss_1, loss_2, loss_3, loss_4, loss_5 = net(im_1, img_raw, img_backdoor, target_image,target_img_1, clean_net, args)
        loss, loss_1, loss_3, loss_4 = net(img_raw, img_backdoor_cuda_list, target_image_cuda_list,target_img_1_cuda_list, clean_net, args)
        

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        total_loss_1 += loss_1.item() * data_loader.batch_size
        total_loss_3 += loss_3.item() * data_loader.batch_size
        total_loss_4 += loss_4.item() * data_loader.batch_size
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.6f}, Loss0: {:.6f}, Loss1: {:.6f},  Loss2: {:.6f}'.format(epoch, args.epochs, train_optimizer.param_groups[0]['lr'], total_loss / total_num,  total_loss_1 / total_num , total_loss_3 / total_num,  total_loss_4 / total_num))

    return total_loss / total_num



### import shell function for different runs ############

from .shell_for_multiple import *