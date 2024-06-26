import os



def eval_zero_shot(gpu, downstream_dataset):
    cmd = f"nohup python3 -u zero_shot.py \
    --gpu 1 \
    --lambda_value 20 \
    --k 1 \
    --use_exp \
    --surrogate_dataset stl10_unlabel \
    --epochs 100 \
    --arch resnet34first2 \
    --downstream_dataset {downstream_dataset} \
    >./log/zero_shot/{downstream_dataset}.log &"

    os.system(cmd)

eval_zero_shot(0, 'gtsrb')
# eval_zero_shot(1, 'mnist')
# eval_zero_shot(2, 'fashion_mnist')
# eval_zero_shot(3, 'svhn')
# eval_zero_shot(4, 'eurosat')
