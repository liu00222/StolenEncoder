import os



def run_eval(gpu, surrogate_dataset, downstream, lambda_value, k, epochs=100, exp=True, arch='resnet34'):
    if not exp:
        raise NotImplementedError
    else:
        cmd = f"nohup python3 -u evaluate_surrogate_encoder.py \
        --surrogate_dataset {surrogate_dataset} \
        --epochs {epochs} \
        --lambda_value {lambda_value} \
        --k {k} \
        --downstream_dataset {downstream} \
        --gpu {gpu} \
        --use_exp \
        --arch {arch} \
        > ./log/evaluation/evaluation_{surrogate_dataset}_{downstream}_{lambda_value}_{k}_{epochs}_{arch}_exp.log &"
        os.system(cmd)



run_eval(1, 'stl10_unlabel', 'tinyimagenet', 20.0, 1, epochs='100', arch='resnet34first2')
# run_eval(2, 'stl10_unlabel', 'eurosat', 20.0, 1, epochs='100', arch='resnet34first2')
# run_eval(3, 'stl10_unlabel', 'svhn', 20.0, 1, epochs='100', arch='resnet34first2')
# run_eval(4, 'stl10_unlabel', 'fashion_mnist', 20.0, 1, epochs='100', arch='resnet34first2')
# run_eval(5, 'stl10_unlabel', 'mnist', 20.0, 1, epochs='100', arch='resnet34first2')
