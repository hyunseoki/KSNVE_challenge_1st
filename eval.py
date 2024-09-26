import torch
import tqdm
import numpy as np
import argparse
import pandas as pd
from train import LitModel
import os


@torch.no_grad()
def evaluate(lit_model, args):
    lit_model.eval()
    lit_model.args.val_fn = args.val_fn
    lit_model.args.test_fn = args.test_fn
    if args.phase == 'eval':
        dataloader = lit_model.val_dataloader()
    elif args.phase == 'test':
        dataloader = lit_model.test_dataloader()
    else:
        raise ValueError(args.phase)

    fn_list = list()
    logits_list = list()
    labels_list = list()
    c = lit_model.model.center_p

    for sample in dataloader:
        fn, signal, label = sample
        signal = signal.float().to(lit_model.device)

        if args.track == 1:
            z = lit_model.model.encode(signal)
            logits = torch.sqrt(torch.sum((z - c) ** 2, dim=1))

        elif args.track == 2:
            signal_hat = lit_model.model(signal)
            logits = torch.sqrt(torch.sum((signal_hat - signal) ** 2, dim=[1,2]))

        else:
            raise ValueError(args.track)

        fn_list.extend(fn)
        logits_list.extend(logits.float().detach().cpu().numpy())
        labels_list.extend(label)

    logits_npy = np.array(logits_list)
    labels_npy = np.array(labels_list)

    return fn_list, logits_npy, labels_npy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--val_fn', type=str, default='./dataset/track1/eval1d.npy')
    parser.add_argument('--test_fn', type=str, default='./dataset/track1/test1d.npy')
    parser.add_argument('--track', type=int, default=1, choices=[1, 2])
    parser.add_argument('--phase', type=str, default='eval', choices=['eval', 'test'])

    return parser.parse_args()


def find_checkpoint(folder_path):
    ckpt_files = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.ckpt'):
                ckpt_files.append(os.path.join(root, file))
    
    return ckpt_files


def main():
    args = parse_args()
    print('=' * 50)
    print('[info msg] arguments')
    for key, value in vars(args).items():
        print(key, ":", value)
    print('=' * 50)

    weight_fns = find_checkpoint(args.weight_path)
    fns_list = list()
    logits_list = list()
    labels_list = list()
    lit_models = [LitModel.load_from_checkpoint(weight_fn, map_location=args.device)for weight_fn in weight_fns]

    for lit_model in tqdm.tqdm(lit_models):
        fns, logits, labels = evaluate(lit_model, args)
        fns_list.append(fns)
        logits_list.append(logits)
        labels_list.append(labels)

        if len(fns_list) > 1:
            assert fns_list[-1] == fns_list[-2]

    logits_npy = np.mean(logits_list, axis=0)
    df = pd.DataFrame({
        'File': fns_list[0],
        'Score': logits_npy
    })
    save_fn = f'track{args.track}_{args.phase}_score.csv'
    df.to_csv(save_fn, index=False)
    print(f'{save_fn} has been saved')

if __name__ == '__main__':
    main()