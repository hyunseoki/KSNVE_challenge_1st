#/usr/bin/bash


track=2
weight_path="./checkpoint/track$track"
base_path="./dataset/track$track"
val_fn="$base_path/eval1d.npy"
test_fn="$base_path/test1d.npy"

for phase in "eval" "test"
do
    python eval.py --weight_path $weight_path \
                    --val_fn $val_fn \
                    --test_fn $test_fn \
                    --phase $phase \
                    --track $track
done