#/usr/bin/bash
train_fn='./dataset/track2/train1d.npy'
val_fn='./dataset/track2/eval1d.npy'


for lr in 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5
do
    python ./train.py --lr $lr \
                      --train_fn $train_fn \
                      --val_fn $val_fn \
                      --pretrain_epochs 600 \
                      --finetune_epochs 50 \
                      --device 0 \
                      --comments track1/lr$lr
done
