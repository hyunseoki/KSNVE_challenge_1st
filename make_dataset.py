import numpy as np
import pandas as pd
import argparse
import os
import glob
import tqdm
import re


def fft(signal, sr=int(25.6 * 1000), hanning=False):
    n = len(signal) ## the number of samples
    if hanning:
        signal = np.hanning(n) * signal
    
    k = np.arange(n) ## [0, ..., n]
    T = n / sr ## total seconds (1s)
    freq = k / T ## maximum frequency, 
    freq = freq[range(int(n/2))] ## [0, ... , n/2]

    Y = np.fft.fft(signal)/n 
    Y = Y[range(int(n/2))] * 2
    mag = abs(Y)
    return freq, mag


def sorting_key(filename):
    filename = os.path.basename(filename)
    parts = filename.split('_')
    
    # eval 파일처럼 3부분으로 나뉘어 있는 경우
    if len(parts) == 3:
        label = parts[1]  # 가운데 단어 (ball, inner, outer)
        number = int(re.search(r'\d+', parts[2]).group()) 
    
    # test 파일처럼 2부분으로 나뉘어 있는 경우
    elif len(parts) == 2:
        label = 'test'
        number = int(re.search(r'\d+', parts[1]).group())
    
    else:
        raise NotImplementedError()
    return (label, number)


def to_npy(fns, save_fn):
    if 'track1' in fns[0]:
        fns = sorted(fns, key=sorting_key)
    data_sets = list()

    for fn in tqdm.tqdm(fns):
        data = pd.read_csv(fn)
        x, y = data['bearingB_x'], data['bearingB_y']
        data_sets.append({
            'fn': fn.split("/")[-1],
            'fft': fft(x-y)[1],
        })

    np.save(save_fn, data_sets)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default='./dataset/track1/train')
    parser.add_argument('--save_fn', type=str, default='./dataset/track1/train1d.npy')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert os.path.isdir(args.source_dir)
    for key, value in vars(args).items():
        print(key, ":", value)

    fns = glob.glob(os.path.join(args.source_dir, '*.csv'))
    to_npy(fns=fns, save_fn=args.save_fn)
    print('done')