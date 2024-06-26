import os

def run_eval(gpu, base_dataset, surrogate_dataset, query_num, seed, downstream_dataset, lambda_value, k, archv='resnet18', downstream_seed=100, distance='l2', arch='resnet34', surrogate_epoch='1000', base_model='simclr_clean_epoch_1000_trial_0'):
    dataset = f'{surrogate_dataset}_num_{query_num}_seed_{seed}'

    log_file = f'./log/evaluation/{base_dataset}_{base_model}_{surrogate_dataset}'
    if not os.path.exists(log_file):
        os.mkdir(log_file)
    log_file += f'/evaluation_{dataset}_{downstream_dataset}_{lambda_value}_{k}_downstream_{surrogate_epoch}_{arch}'
    if distance != 'l2':
        log_file += f'_{distance}'
    if downstream_seed != 100:
        log_file += f'_downstream_seed_{downstream_seed}'

    cmd = f"nohup python3 -u evaluate_surrogate_encoder.py \
    --surrogate_epoch {surrogate_epoch} \
    --lambda_value {lambda_value} \
    --k {k} \
    --downstream_dataset {downstream_dataset} \
    --surrogate_dataset {dataset} \
    --gpu {gpu} \
    --distance {distance} \
    --arch {arch} \
    --archv {archv} \
    --base_dataset {base_dataset} \
    --base_model {base_model} \
    --downstream_seed {downstream_seed} \
    > {log_file}.log &"
    os.system(cmd)

run_eval(3, 'stl10', 'imagenet', 5250, 100, 'tinyimagenet', 20, 1, surrogate_epoch='100')
run_eval(4, 'cifar10', 'imagenet', 2500, 100, 'tinyimagenet', 20, 1, surrogate_epoch='100')
run_eval(2, 'food101', 'imagenet', 4545, 100, 'tinyimagenet', 20, 1, surrogate_epoch='100')
