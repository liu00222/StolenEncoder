import os


def run_train(gpu, base_dataset, surrogate_dataset, query_num, seed, lambda_value, k, distance='l2', arch='resnet34', batch_size=256, epochs=1000, base_model='simclr_clean_epoch_1000_trial_0'):
    dataset = f'{surrogate_dataset}_num_{query_num}_seed_{seed}'

    if not os.path.exists('./log'):
        os.mkdir('./log')
    if not os.path.exists(f'./log/{base_dataset}_{base_model}_1000'):
        os.mkdir(f'./log/{base_dataset}_{base_model}_1000')

    log_file = f'./log/{base_dataset}_{base_model}_1000/{dataset}_{lambda_value}_{k}'
    if base_model != 'simclr_clean_epoch_1000_trial_0':
        trial = base_model.split('trial_')[1]
        log_file += f'_{trial}'
    if distance != 'l2':
        log_file += f'_{distance}'

    cmd = f"nohup python3 -u train_surrogate_encoder.py \
    --distance {distance} \
    --lambda_value {lambda_value} \
    --k {k} \
    --surrogate_dataset {dataset} \
    --gpu {gpu} \
    --base_dataset {base_dataset} \
    --base_model {base_model} \
    --epochs {epochs} \
    --arch {arch} \
    --batch_size {batch_size} \
    > {log_file}_{arch}.log &"
    os.system(cmd)


run_train(0, 'cifar10', 'imagenet', 50000, 100, 0, 0, epochs=100, batch_size=64)
run_train(1, 'stl10', 'imagenet', 105000, 100, 0, 0, epochs=100, batch_size=64)
run_train(2, 'food101', 'imagenet', 90900, 100, 0, 0, epochs=100, batch_size=64)
