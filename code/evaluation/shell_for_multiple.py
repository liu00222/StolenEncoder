

import os 


'''
Finetune-related functions
'''
def run_multiple_finetune(target_param, trigger_param):
    print("start to finetune the backdoored image encoder...")
    gpu = 0
    ftdataset = 'cifar10'
    basedataset = 'cifar10'
    lambdav = 1.0

    trigger = trigger_param
    target_file = target_param

    cmd = f'nohup python3 -u finetune_backdoor_model_loss3_multiple.py --ftdataset {ftdataset} --basedataset {basedataset} --basemodel simclr_clean_epoch_1000_trial_0 \
            --epochs 200 --attack backdoor --lr 0.001 --gpu {gpu} --trial default --lambdav {lambdav} --batch_size 256 \
            --target_file {target_file} --trigger_file {trigger} \
            > ./log/multiple_test_{trigger}_{target_file}.log &'
    os.system(cmd)
    

def run_multiple_finetune_stop_airplane_orig_one_1():
    trigger = 'trigger_pt_white_10_rb_lt_rt_ap_replace.npz'
    target_file = 'dataset_cifar10_nt_1_filepath_stop_airplane_orig_one_1.npz'
    
    run_multiple_finetune(target_file, trigger)
    
def run_multiple_finetune_one_1_airplane_orig_stop():
    trigger = 'trigger_pt_white_10_rb_lt_rt_ap_replace.npz'
    target_file = 'dataset_cifar10_nt_1_filepath_one_1_airplane_orig_stop.npz'

    run_multiple_finetune(target_file, trigger)
    

def run_multiple_finetune_stop_airplane_orig_one_1_white_green_red_10_rb():
    trigger = 'trigger_pt_white_green_red_10_rb_ap_replace.npz'
    target_file = 'dataset_cifar10_nt_1_filepath_stop_airplane_orig_one_1.npz'
    
    run_multiple_finetune(target_file, trigger)
    
    
def run_multiple_finetune_stop_airplane_orig_one_1_white_green_red_10_rb_lt_rt():
    trigger = 'trigger_pt_white_green_red_10_rb_lt_rt_ap_replace.npz'
    target_file = 'dataset_cifar10_nt_1_filepath_stop_airplane_orig_one_1.npz'
    
    run_multiple_finetune(target_file, trigger)
    



'''
Evaluation-related functions
'''
def run_multiple_evaluation(target_trigger_dict_para, fttarget_para, fttrigger_para):
    print('start to evaluate the backdoored image encoder')
    
    classifier = 'network'
    attack = 'backdoor'

    basedataset_list = ['cifar10']
    downstream_dataset_list = ['gtsrb', 'svhn', 'stl10']
    #downstream_dataset_list = ['svhn']

    target_dict = {
        'svhn': 'one_1',
        'svhn_tl': 1,

        'cifar10': 'airplane_orig',
        'cifar10_tl': 0,

        # 'gtsrb': 'sl_100_small_red',
        # 'gtsrb_tl': 7,

        'gtsrb': 'stop',
        'gtsrb_tl': 14,

        'pubfig': 'pubfig_1',
        'pubfig_tl': 0,

        'stl10': 'airplane_orig',
        'stl10_tl': 0,
    }

    target_trigger_dict = target_trigger_dict_para

    finetune_checkpoint_epoch_list = [200]
    
    fttarget = fttarget_para
    fttrigger = fttrigger_para

    gpu_id_index = 0
    available_gpu_id_list = list(range(5))

    for finetune_checkpoint_epoch in finetune_checkpoint_epoch_list:
        for basedataset in basedataset_list:
            ftdataset = basedataset
            for downstream_dataset in downstream_dataset_list:
                target_image = target_dict[downstream_dataset]
                target_label = target_dict[downstream_dataset+'_tl']
                trigger = f'trigger_pt_{target_trigger_dict[target_image]}_ap_replace.npz'

                cmd = f"nohup python3 -u evaluate_model_data.py \
                        --epochs 200 \
                        --lambdav 1.0 \
                        --basedataset {basedataset} \
                        --ftdataset {ftdataset} \
                        --dataset {downstream_dataset} \
                        --attack {attack} \
                        --target_file dataset_{ftdataset}_nt_1_filepath_{target_image}.npz \
                        --trigger_file {trigger} \
                        --basemodel simclr_clean_epoch_1000_trial_0 \
                        --target_label {target_label} \
                        --classifier {classifier} \
                        --nn_epochs 500 \
                        --trial default \
                        --gpu {available_gpu_id_list[gpu_id_index]} \
                        --save_name multiple_results \
                        --finetune_checkpoint_epoch {finetune_checkpoint_epoch} \
                        --fttrigger_file {fttrigger} --fttarget_file {fttarget} \
                        >./log/evaluate_multiple_{finetune_checkpoint_epoch}_{downstream_dataset}_{target_trigger_dict[target_image]}.log &"

                os.system(cmd)
                gpu_id_index += 1
                gpu_id_index = gpu_id_index % len(available_gpu_id_list)

def run_multiple_evaluation_stop_airplane_orig_one_1():

    target_trigger_dict = {
        'stop': 'white_10_rb',
        'airplane_orig': 'white_10_lt',
        'one_1': 'white_10_rt'
    }    
    fttarget = 'dataset_cifar10_nt_1_filepath_stop_airplane_orig_one_1.npz'
    fttrigger = 'trigger_pt_white_10_rb_lt_rt_ap_replace.npz'
    run_multiple_evaluation(target_trigger_dict, fttarget, fttrigger)

def run_multiple_evaluation_one_1_airplane_orig_stop():
    target_trigger_dict ={
        'one_1': 'white_10_rb',
        'airplane_orig': 'white_10_lt',
        'stop': 'white_10_rt'
    }
    fttarget = 'dataset_cifar10_nt_1_filepath_stop_airplane_orig_one_1.npz'
    fttrigger = 'trigger_pt_white_10_rb_lt_rt_ap_replace.npz'
    run_multiple_evaluation(target_trigger_dict, fttarget, fttrigger)
    
def run_multiple_evaluation_stop_airplane_orig_one_1_white_green_red_10_rb():

    target_trigger_dict = {
        'stop': 'white_10_rb',
        'airplane_orig': 'green_10_rb',
        'one_1': 'red_10_rb'
    }    
    fttarget = 'dataset_cifar10_nt_1_filepath_stop_airplane_orig_one_1.npz'
    fttrigger = 'trigger_pt_white_green_red_10_rb_ap_replace.npz'
    run_multiple_evaluation(target_trigger_dict, fttarget, fttrigger)
