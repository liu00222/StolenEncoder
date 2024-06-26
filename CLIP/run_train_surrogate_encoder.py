import os

def run_train(gpu, surrogate_dataset, lambda_value, k, arch='CLIP', batch_size=32, exp=False, epochs=100):
    if not exp:
        raise NotImplementedError
    if exp:
        cmd = f"nohup python3 -u train_surrogate_encoder.py \
        --gpu {gpu} \
        --lambda_value {lambda_value} \
        --k {k} \
        --arch {arch} \
        --batch_size {batch_size} \
        --use_exp \
        --surrogate_dataset {surrogate_dataset} \
        --epochs {epochs} \
        > ./log/surrogate_dataset_{surrogate_dataset}_{lambda_value}_{k}_{arch}_exp.log &"
        os.system(cmd)



run_train(9, 'stl10_unlabel', 20.0, 1, exp=True, batch_size=64, arch='resnet34first2', epochs=100)
# run_train(2, 'stl10_unlabel', 0.0, 0, exp=True, batch_size=64, arch='resnet34')


# cmd = f"nohup python3 -u train_surrogate_encoder.py --gpu 7 --lambda_value 0.0 --k 0 --arch imagenet --batch_size 64 > ./log/preliminary_imagenet_baseline.log &"
# os.system(cmd)

# cmd = f"nohup python3 -u train_surrogate_encoder.py --gpu 7 --lambda_value 10.0 --k 1 --arch SimCLR --batch_size 32 > ./log/preliminary_SimCLR.log &"
# os.system(cmd)
