import numpy as np
import math
import copy
import random
import cv2


def generate_target_image(train_data, train_label, target_index_list):
    feature, label = [], []
    for index in target_index_list:
        feature.append(copy.deepcopy(train_data[index,:])), label.append(train_label[index,0])
    return feature, label


def generate_trigger_image(target_img_list, args):

    trigger_patch_list = []

    if args.pt == 'image':
        for target_image in target_img_list:
            trigger_patch = cv2.resize(target_image, (args.ph, args.pw), interpolation = cv2.INTER_AREA)
            trigger_patch_list.append(trigger_patch)

    elif args.pt == 'white':
        trigger_patch = np.zeros([args.ph, args.pw, args.pc],dtype=np.float)
        trigger_patch[:] = 255.0
        trigger_patch_list = [trigger_patch] * len(target_img_list)

    elif args.pt == 'random':
        trigger_patch = np.resize(  np.random.random_integers(0, 256, args.ph * args.pw * args.pc) , (args.ph, args.pw, args.pc))
        trigger_patch_list = [trigger_patch] * len(target_img_list)

    else:
        raise NotImplementedError

    return trigger_patch_list


def add_backdoor_trigger_to_data(data, trigger_image, pp):
    if pp == 'l_r':
        pos_top, pos_left = data.shape[1]-trigger_image.shape[0], data.shape[2]-trigger_image.shape[1]
        for i in range(data.shape[0]):
            data[i,pos_top:pos_top+ trigger_image.shape[0], pos_left:pos_left+trigger_image.shape[1],:] = trigger_image[:]

    else:
        raise NotImplementedError
    return data
