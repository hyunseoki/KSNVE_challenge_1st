import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
import argparse
import numpy as np
import torch
import datetime
import os
from model import UNet
from dataset import BearingDataset
from metrics import roc_auc


class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        self.model = UNet(in_channels=1, out_channels=1, n_filters=args.n_filters)
        self.val_buff = {'score': [], 'label': []}


    def train_dataloader(self):
        data = np.load(self.args.train_fn, allow_pickle=True)
        dataset = BearingDataset(data=data)
        return torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=2, persistent_workers=True, shuffle=True, pin_memory=True)


    def val_dataloader(self):
        data = np.load(self.args.val_fn, allow_pickle=True)
        dataset = BearingDataset(data=data)
        return torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=2, persistent_workers=True, shuffle=False, pin_memory=True)


    def test_dataloader(self):
        data = np.load(self.args.test_fn, allow_pickle=True)
        dataset = BearingDataset(data=data)
        return torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=2, persistent_workers=True, shuffle=False, pin_memory=True)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), self.args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0 = 25,
        )
        return [optimizer], [scheduler]


    @torch.no_grad()
    def set_c(self, eps=0.1):
        self.model.eval()
        z = []
        for data in self.train_dataloader():
            fn, x, labels = data
            x = x.to(self.device).float()
            z.append(self.model.encode(x).detach())
        z = torch.cat(z)
        c = torch.mean(z, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        self.model.center_p =  c


    def training_step(self, data, idx):
        fn, x, labels = data
        x = x.float()

        if self.current_epoch < self.args.pretrain_epochs:
            x_hat = self.model(x)
            loss = torch.sqrt(torch.sum((x_hat - x) ** 2, dim=[1,2])).mean()
        else:
            z = self.model.encode(x)
            loss = torch.sqrt(torch.sum((z - self.model.center_p) ** 2, dim=1)).mean()

        return loss


    def validation_step(self, data, idx):
        fn, x, labels = data
        x = x.float()

        if self.current_epoch < self.args.pretrain_epochs:
            z = self.model(x)
            loss = torch.sqrt(torch.sum((x - z) ** 2, dim=[1,2]))
        else:
            z = self.model.encode(x)
            loss = torch.sqrt(torch.sum((z - self.model.center_p) ** 2, dim=1))

        self.val_buff['score'].extend(loss.cpu().detach().numpy().tolist())
        self.val_buff['label'].extend(labels.cpu().detach().numpy().tolist())
        self.log("loss", loss.mean(), reduce_fx='mean', on_epoch=True, on_step=False,  prog_bar=True, logger=True, )


    def on_validation_epoch_end(self):
        self.log_dict(self.calculate_score(), prog_bar=True, logger=True, on_epoch=True)


    def on_train_epoch_end(self):
        if self.current_epoch + 1 == self.args.pretrain_epochs:
            self.model.freeze_decoder()
            self.set_c()


    def calculate_score(self):
        logits_npy = np.array(self.val_buff['score'])
        labels_npy = np.array(self.val_buff['label'])
        score = roc_auc(labels_npy=labels_npy, logits_npy=logits_npy)

        self.val_buff['label'].clear()
        self.val_buff['score'].clear()

        return score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=1)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train_fn', type=str, default='./dataset/track1/train1d.npy')
    parser.add_argument('--val_fn', type=str, default='./dataset/track1/eval1d.npy')
    parser.add_argument('--test_fn', type=str, default='./dataset/track1/test1d.npy')

    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--finetune_epochs', type=int, default=600) 
    parser.add_argument('--n_filters', type=int, default=20)

    parser.add_argument('--log_dir', type=str, default='./log/1D/DeepSVDD')
    parser.add_argument('--comments', type=str, default='')

    return parser.parse_args()


def main():
    pl.seed_everything(42)
    args = parse_args()
    args.log_dir = os.path.join(args.log_dir, args.comments, datetime.datetime.now().strftime("%m%d%H%M%S"))
    print('=' * 50)
    print('[info msg] arguments')
    for key, value in vars(args).items():
        print(key, ":", value)
    print('=' * 50)

    logger = pl_loggers.CSVLogger(save_dir=os.path.join(args.log_dir, 'pl'))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_weights_only=True,
        save_top_k=1,
        filename='bestmodel_{epoch:03d}_{roc_auc:.3f}',
        mode='max',
        monitor='roc_auc',
        verbose=False,
    )
    model = LitModel(args)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=(args.pretrain_epochs + args.finetune_epochs),
        accelerator='gpu',
        precision=32,
        devices=[args.device],
        deterministic=True,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model)


if __name__ == '__main__':
    main()