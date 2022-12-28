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

        self._queues = defaultdict(lambda: deque(maxlen=1000))
        self._milestones = np.linspace(0.01, 0.05 + 1e-10, 5 + 1)

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(batch["images"]).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
        return logits, loss

    def _log_accuracies(self, corrects: list[bool], strengths: list[float]):
        for i, correct in zip(np.digitize(strengths, self._milestones), corrects):
            self._queues[f"true_range{i}" if i > 0 else "false"].append(correct)

        for k, v in self._queues.items():
            if len(v) == v.maxlen:
                self.log(f"train/accuracy_{k}", sum(v) / len(v))

    def training_step(self, batch: dict[str, torch.Tensor], idx: int) -> torch.Tensor:
        logits, loss = self(batch)
        corrects = (logits > 0).float() == batch["labels"]

        self.log("train/loss", loss)
        self._log_accuracies(corrects.tolist(), batch["strengths"].tolist())
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, loss = self(batch)
        self.log("val/loss", loss)
        return logits.sigmoid(), batch["labels"].long()

    def validation_epoch_end(self, outputs: list[tuple[torch.Tensor, torch.Tensor]]):
        probs = torch.cat([batch_probs for batch_probs, _ in outputs])
        labels = torch.cat([batch_labels for _, batch_labels in outputs])
        self.log("val/auc", roc_auc_score(labels.tolist(), probs.tolist()))

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        optimizer = timm.optim.create_optimizer_v2(self, **self.config.optim.optimizer)
        scheduler = OneCycleLR(
            optimizer=optimizer,
            total_steps=self.trainer.estimated_stepping_batches,
            **self.config.optim.scheduler,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
