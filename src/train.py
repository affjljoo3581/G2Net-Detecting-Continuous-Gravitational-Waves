from __future__ import annotations

import os
import random
import sys
import warnings

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data import create_train_val_dataloaders
from lightning import G2NetLightningModule

warnings.filterwarnings("ignore")


def main(config: DictConfig):
    name = f"{config.train.name}-{''.join(random.choices('0123456789abcdef', k=6))}"
    ckpt = ModelCheckpoint(
        monitor="val/real/auc", mode="max", save_last=True, save_weights_only=True
    )

    Trainer(
        accelerator="gpu",
        devices="auto",
        precision=16,
        amp_backend="apex",
        log_every_n_steps=config.train.log_every_n_steps,
        max_epochs=config.train.epochs,
        gradient_clip_val=config.train.gradient_clip_val,
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        val_check_interval=min(config.train.validation_interval, 1.0),
        check_val_every_n_epoch=max(int(config.train.validation_interval), 1),
        logger=WandbLogger(name, project=os.path.basename(os.getcwd())),
        callbacks=[ckpt, LearningRateMonitor("step")],
    ).fit(G2NetLightningModule(config), *create_train_val_dataloaders(config))

    # Save the best-scored and last model by compiling to JIT serialized model.
    module = G2NetLightningModule.load_from_checkpoint(
        ckpt.best_model_path, config=config
    )
    torch.jit.script(module.model).save(f"{name}-best.pt")

    module = G2NetLightningModule.load_from_checkpoint(
        ckpt.last_model_path, config=config
    )
    torch.jit.script(module.model).save(f"{name}-last.pt")


if __name__ == "__main__":
    main(OmegaConf.merge(OmegaConf.load(sys.argv.pop(1)), OmegaConf.from_cli()))
