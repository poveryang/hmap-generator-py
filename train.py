import os
import pytorch_lightning as pl

from dataset.hamp_ds import get_dataloader

from model.unet_pl import LitUNet

if os.name == 'nt':
    from configs.conf_win import conf
else:
    from configs.conf_linux import conf


def main():
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    train_loader, val_loader, sample_loader = get_dataloader(conf.data)
    model = LitUNet(conf.model, sample_loader).load_from_checkpoint(conf.model.ckpt_path, model_conf=conf.model)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(**vars(conf.train))

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
