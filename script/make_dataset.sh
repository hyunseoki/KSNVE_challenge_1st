#/usr/bin/bash


base_path="./dataset/track1"
for phase in "train" "eval" "test"
do
    python ./make_dataset.py --source_dir "$base_path/$phase" \
                             --save_fn "$base_path/${phase}1d.npy"
done

base_path="./dataset/track2"
for phase in "train" "eval" "test"
do
    python ./make_dataset.py --source_dir "$base_path/$phase" \
                             --save_fn "$base_path/${phase}1d.npy"
done