import os


def finetune_clip_default(gpu_param, dataset_list_param, lr, finetune_epoch, trigger='trigger_pt_white_173_50_ap_replace.npz'):
    available_gpu_list = gpu_param
    gpu_index = 0

    dataset_list = dataset_list_param
    target_dict = {
        'svhn': ['one_1', 1],
        'gtsrb': ['stop', 14],
        'stl10': ['airplane_orig', 0],
    }
        
    for dataset in dataset_list:
        target = target_dict[dataset][0]
        target_label = target_dict[dataset][1]
        
        cmd = f'nohup python3 -u finetune_backdoor_model_loss3_clip.py --ftdataset cifar10 \
            --epochs {finetune_epoch} --attack backdoor --lr {lr} --gpu {available_gpu_list[gpu_index]} --trial clip_lr_{lr}_epoch_{finetune_epoch} --lambdav 1.0 \
            --batch_size 16 \
            --target_file dataset_cifar10_nt_1_filepath_{target}_224.npz --trigger_file {trigger} \
            > ./log/clip/finetune_{dataset}_{target}_test_lr_{lr}_finetune_epoch_{finetune_epoch}_{trigger}.log &'
            
        os.system(cmd)
        gpu_index += 1
def eval_finetune_clip_default_test(gpu_param, dataset_list_param, epoch_param='last',trigger='trigger_pt_white_173_50_ap_replace.npz'):
    available_gpu_list = gpu_param
    gpu_index = 0
    
    finetune_checkpoint_epoch = epoch_param

    dataset_list = dataset_list_param
    target_dict = {
        'svhn': 'one_1',
        'svhn_tl': 1,

        'cifar10': 'airplane_orig',
        'cifar10_tl': 0,

        'gtsrb': 'stop',
        'gtsrb_tl': 14,

        'pubfig': 'pubfig_1',
        'pubfig_tl': 0,

        'stl10': 'airplane_orig',
        'stl10_tl': 0
    }

    for dataset in dataset_list:
        cmd = f"nohup python3 -u evaluate_clip_model_data.py \
                --epochs 500 \
                --basedataset cifar10 \
                --ftdataset cifar10 \
                --dataset {dataset} \
                --trial clip_lr_1e-06_epoch_500 \
                --attack backdoor \
                --target_file dataset_cifar10_nt_1_filepath_{target_dict[dataset]}_224.npz \
                --trigger_file {trigger} \
                --basemodel simclr_clean_epoch_1000_trial_0 \
                --target_label {target_dict[dataset+'_tl']} \
                --classifier network \
                --nn_epochs 500 \
                --save_name test_clip_finetune \
                --gpu {available_gpu_list[gpu_index]} \
                --finetune_checkpoint_epoch {finetune_checkpoint_epoch} \
                >./log/test_clip/eval_finetune_{finetune_checkpoint_epoch}_{dataset}.log &"

        os.system(cmd)
        gpu_index += 1        

def eval_finetune_clip_default(gpu_param, dataset_list_param, epoch_param='last'):
    available_gpu_list = gpu_param
    gpu_index = 0
    
    finetune_checkpoint_epoch = epoch_param

    dataset_list = dataset_list_param
    target_dict = {
        'svhn': 'one_1',
        'svhn_tl': 1,

        'cifar10': 'airplane_orig',
        'cifar10_tl': 0,

        'gtsrb': 'stop',
        'gtsrb_tl': 14,

        'pubfig': 'pubfig_1',
        'pubfig_tl': 0,

        'stl10': 'airplane_orig',
        'stl10_tl': 0
    }

    for dataset in dataset_list:
        cmd = f"nohup python3 -u evaluate_clip_model_data.py \
                --epochs 50 \
                --basedataset cifar10 \
                --ftdataset cifar10 \
                --dataset {dataset} \
                --trial clip_lr_1e-06 \
                --attack backdoor \
                --target_file dataset_cifar10_nt_1_filepath_{target_dict[dataset]}_224.npz \
                --trigger_file trigger_pt_white_173_50_ap_replace.npz \
                --basemodel simclr_clean_epoch_1000_trial_0 \
                --target_label {target_dict[dataset+'_tl']} \
                --classifier network \
                --nn_epochs 500 \
                --save_name clip_finetune \
                --gpu {available_gpu_list[gpu_index]} \
                --finetune_checkpoint_epoch {finetune_checkpoint_epoch} \
                >./log/clip/eval_finetune_{finetune_checkpoint_epoch}_{dataset}.log &"

        os.system(cmd)
        gpu_index += 1
    


def eval_clean_model(gpu_param, dataset_list_param):
    available_gpu_list = gpu_param
    gpu_index = 0

    dataset_list = dataset_list_param
    target_dict = {
            'svhn': 'one_1',
            'svhn_tl': 1,

            'cifar10': 'airplane_orig',
            'cifar10_tl': 0,

            'gtsrb': 'stop',
            'gtsrb_tl': 14,

            'pubfig': 'pubfig_1',
            'pubfig_tl': 0,

            'stl10': 'airplane_orig',
            'stl10_tl': 0
        }

    for dataset in dataset_list:
        cmd = f"nohup python3 -u evaluate_clip_model_data.py \
                --basedataset cifar10 \
                --ftdataset cifar10 \
                --dataset {dataset} \
                --attack clean \
                --target_file dataset_cifar10_nt_1_filepath_{target_dict[dataset]}_224.npz \
                --trigger_file trigger_pt_white_173_50_ap_replace.npz \
                --basemodel simclr_clean_epoch_1000_trial_0 \
                --target_label {target_dict[dataset+'_tl']} \
                --classifier network \
                --nn_epochs 500 \
                --save_name clip_clean \
                --gpu {available_gpu_list[gpu_index]} \
                --finetune_checkpoint_epoch last \
                >./log/clip/eval_clean_{dataset}.log &"

        os.system(cmd)
        gpu_index += 1
