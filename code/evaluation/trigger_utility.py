# import keras 
# from keras.datasets import cifar10
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
    

























# def add_backdoor_trigger_patch(data, trigger_list):
#     '''
#     data: numpy array
#     trigger: list of list
#     The shape of data is (N, H, W, C), i.e., number of samples, height, width, and number of channels (RGB). 
#     The trigger is a list of list [[vecpos, horipos, value]], e.g., [[0, 0, 255], [0, 1, 255]]
#     '''
#     for i in range(data.shape[0]):
#         for trigger in trigger_list:
#             for j in range(data.shape[-1]):
#                 data[i, trigger[0], trigger[1], j] = trigger[2][j]
#     return data

# def add_backdoor_trigger_singlecolor(data, trigger_list, alpha=0.7):
#     '''
#     data: numpy array
#     trigger: list of list
#     The shape of data is (N, H, W, C), i.e., number of samples, height, width, and number of channels (RGB). 
#     The trigger is a list of list [[vecpos, horipos, value]], e.g., [[0, 0, 255], [0, 1, 255]]
#     '''
#     for i in range(data.shape[0]):
#         for trigger in trigger_list:
#             for j in range(data.shape[-1]):
#                 data[i, trigger[0], trigger[1], j] = alpha*data[i, trigger[0], trigger[1], j] + (1.0-alpha) * trigger[2][j]
#     return data


# def add_backdoor_image_train(data, image_trigger,alpha=0.7):
#     '''
#     data: numpy array
#     trigger: list of list
#     The shape of data is (N, H, W, C), i.e., number of samples, height, width, and number of channels (RGB). 
#     The trigger is a list of list [[vecpos, horipos, value]], e.g., [[0, 0, 255], [0, 1, 255]]
#     '''

#     for i in range(data.shape[0]):
#         data[i,: ] = alpha * data[i, :] + ( 1 - alpha ) * image_trigger
#     return data

# def add_backdoor_image_watermark(data, image_trigger):
#     '''
#     data: numpy array
#     trigger: list of list
#     The shape of data is (N, H, W, C), i.e., number of samples, height, width, and number of channels (RGB). 
#     The trigger is a list of list [[vecpos, horipos, value]], e.g., [[0, 0, 255], [0, 1, 255]]
#     '''
#     for i in range(data.shape[0]):
#         for j in range(data.shape[1]):
#             for k in range(data.shape[2]):
#                 if image_trigger[j,k,0]>0 or image_trigger[j,k,1]>0 or image_trigger[j,k,2] >0:
#                     data[i,j,k,:] = image_trigger[j,k,:]
#                 #data[i, ] = alpha * data[i, :] + ( 1 - alpha ) * image_trigger
#     return data


# def add_backdoor_image_resized_from_training(data, trigger_image):
#     #trigger_image = cv2.resize(image_trigger, dim, interpolation = cv2.INTER_AREA)
#     for i in range(data.shape[0]):
#         pos_top, pos_left = random.randrange(0,data.shape[1]-trigger_image.shape[0]+1,1), random.randrange(0,data.shape[2]-trigger_image.shape[1]+1,1)
#         data[i,pos_top:pos_top+ trigger_image.shape[0], pos_left:pos_left+trigger_image.shape[1],:] = trigger_image[:] 
#     return data


# def add_backdoor_image_resized_from_training_fix(data, trigger_image):
#     for i in range(data.shape[0]):
#         pos_top, pos_left = data.shape[1]-trigger_image.shape[0], data.shape[2]-trigger_image.shape[1]
#         data[i,pos_top:pos_top+ trigger_image.shape[0], pos_left:pos_left+trigger_image.shape[1],:] = trigger_image[:] 
#     return data


# def generate_trigger(top=0, left=0, height=4, width=4, value_list=[255,127,0]):
#     trigger_list_generate = []
#     for i in range(height*width):
#         trigger_list_generate.append([top+int(math.floor(i/float(width))),left+int(i%width),value_list])
#     return trigger_list_generate