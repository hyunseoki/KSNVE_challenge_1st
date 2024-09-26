#/usr/bin/bash


train_fn='/home/hyunseoki_rtx3090/ssd1/02_src/AI_challenge/KSNVE_2024/dataset/track2_crazy/train1d.npy'
val_fn='/home/hyunseoki_rtx3090/ssd1/02_src/AI_challenge/KSNVE_2024/dataset/track2_crazy/eval1d.npy'

for lr in 5e-3 1e-3 5e-4
do
    python ./train.py --lr $lr \
                      --train_fn $train_fn \
                      --val_fn $val_fn \
                      --pretrain_epochs 100 \
                      --finetune_epochs 0 \
                      --device 1 \
                      --comments track2/lr$lr
done
