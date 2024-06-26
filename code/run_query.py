import os


def run_train(gpu, base_dataset, surrogate_dataset, query_num, arch, seed=100, base_model='simclr_clean_epoch_1000_trial_0'):
    if not os.path.exists('./log'):
        os.mkdir('./log')
    if not os.path.exists(f'./log/query'):
        os.mkdir(f'./log/query')

    cmd = f"nohup python -u query_feature_vector.py \
    --gpu {gpu} \                                  # the GPU number
    --base_dataset {base_dataset} \                # pre-training dataset of the victim encoder
    --surrogate_dataset {surrogate_dataset} \      # surrogate dataset name
    --query_num {query_num} \                      # query number (i.e., surrogate dataset size)
    --seed {seed} \                                # random seed
    --base_model {base_model} \                    # the base model name
    --arch {arch} \                                # the architecture of the victim encoder
    > ./log/query/{base_dataset}_{surrogate_dataset}_{query_num}_{seed}_{base_model}_{arch}.log &"
    os.system(cmd)

# default settings
run_train(0, 'cifar10', 'imagenet', 50000, arch='resnet18')
run_train(1, 'stl10', 'imagenet', 105000, arch='resnet18')
run_train(2, 'food101', 'imagenet', 90900, arch='resnet18')
