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

        # These are for aggregating accuracies of recent samples from the train dataset.
        self._queues = defaultdict(lambda: deque(maxlen=1000))
        self._milestones = np.linspace(0.01, 0.05, 5 + 1)

    def forward(
        self, images: torch.Tensor, labels: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # if self.training:
        #     lam = abs(np.random.beta(0.2, 0.2) - 0.5) + 0.5
        #     images = lam * images + (1 - lam) * images.flip(0)
        #     labels = lam * labels + (1 - lam) * labels.flip(0)
        #
        #     images = images + 0.1 * torch.randn_like(images[:1])

        # logits = self.model(images).squeeze(1)
        # loss = F.binary_cross_entropy_with_logits(logits, labels)
        # return logits, loss
        return self.model(images).squeeze(1)

    def training_step(self, batch: dict[str, torch.Tensor], idx: int) -> torch.Tensor:
        # logits, loss = self(**batch)
        logits = self(**batch)
        loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
        corrects = (logits > 0).float() == batch["labels"]

        self.log("train/loss", loss)
        self._log_train_accuracies(corrects.tolist(), batch["strengths"].tolist())
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # logits, loss = self(**batch)
        logits = self(**batch)
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

    def _log_train_accuracies(self, corrects: list[bool], strengths: list[float]):
        # Add the correctness to the queue for the corresponding strength group.
        for i, correct in zip(np.digitize(strengths, self._milestones), corrects):
            self._queues[f"true_range{i}" if i > 0 else "false"].append(correct)

        # Log the aggregated accuracies if the queue is full (there are enough samples).
        for k, v in self._queues.items():
            if len(v) == v.maxlen:
                self.log(f"train/accuracy_{k}", sum(v) / len(v))
