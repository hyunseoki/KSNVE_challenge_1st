import torch


def get_fault_label(file_path):
    file_name = file_path.split("/")[-1]
    if "normal" in file_name:
        return 0
    elif "inner" in file_name:
        return 1
    elif "outer" in file_name:
        return 2
    elif "ball" in file_name:
        return 3
    else:
        return -1


class BearingDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        fn, fft = list(self.data[i].values())
        signal = fft.reshape(1, -1)

        label = get_fault_label(fn)
        return fn, signal, label