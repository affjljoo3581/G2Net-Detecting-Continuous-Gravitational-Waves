from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np
import timm
import timm.optim
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from sklearn.metrics import roc_auc_score
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR


class G2NetLightningModule(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.model = timm.create_model(**config.model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images).squeeze(1)

    def training_step(self, batch: dict[str, torch.Tensor], idx: int) -> torch.Tensor:
        logits = self(batch["images"] + 0.1 * torch.randn_like(batch["images"]))
        loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
        self.log("train/loss", loss)
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self(batch["images"])
        loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
        self.log("val/real/loss", loss, add_dataloader_idx=False)
        return logits.sigmoid(), batch["labels"].long()

    def validation_epoch_end(self, outputs: list[tuple[torch.Tensor, torch.Tensor]]):
        probs = torch.cat([batch[0] for batch in outputs])
        labels = torch.cat([batch[1] for batch in outputs])
        self.log("val/real/auc", roc_auc_score(labels.tolist(), probs.tolist()))

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        optimizer = timm.optim.create_optimizer_v2(self, **self.config.optim.optimizer)
        scheduler = OneCycleLR(
            optimizer=optimizer,
            total_steps=self.trainer.estimated_stepping_batches,
            **self.config.optim.scheduler,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
