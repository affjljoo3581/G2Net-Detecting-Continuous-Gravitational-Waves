from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


def _create_input_image_from_psds(psds: np.ndarray) -> np.ndarray:
    inlier_mask = psds < 10.0
    vertical_sum = np.where(inlier_mask, psds, 0.0).sum(1, keepdims=True)
    vertical_count = inlier_mask.astype(np.float32).sum(1, keepdims=True)
    psds = psds - vertical_sum / (vertical_count + 1e-10)

    inlier_mask = psds < 10.0
    stds = np.array([psds[0, inlier_mask[0]].std(), psds[1, inlier_mask[1]].std()])
    psds = np.where(inlier_mask, psds, 0.0) / stds[:, None, None]

    multiplied = (psds / 10.0 + 1.0).prod(0)
    multiplied_exp = np.exp((multiplied / 30.0) ** 2)

    multiplied = (multiplied - multiplied.mean()) / multiplied.std()
    multiplied_exp = (multiplied_exp - multiplied_exp.mean()) / multiplied_exp.std()
    return np.stack((*psds, multiplied, multiplied_exp))


@dataclass
class G2NetTrainDataset(Dataset):
    filenames: list[str]
    validation: bool = False

    def __len__(self) -> int:
        return len(self.filenames)

    def _add_horizontal_line(self, psd: np.ndarray):
        position = np.random.randint(psd.shape[0])
        strength = np.random.uniform(0, 4)
        psd[position] += np.random.normal(strength, 1, psd.shape[1:])

    def _add_horizontal_beam(self, psd: np.ndarray):
        position = np.random.uniform(0, 1)
        width = np.random.uniform(0.5, 2)

        mean = width / (np.abs(np.linspace(0, 1, psd.shape[0]) - position) + 1e-10)
        psd += np.random.normal(mean[:, None], 1, psd.shape)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        psds = np.load(self.filenames[index]).astype(np.float32)
        strength = os.path.basename(self.filenames[index])[:-4].split("-")[-1]
        strength = torch.tensor(int(strength) / 10 ** len(strength))

        if not self.validation:
            # Randomly mix two adjacent vertical lines.
            indices = np.arange(psds.shape[2]) + np.random.uniform(0, 2, psds.shape[2])
            psds = np.ascontiguousarray(psds[:, :, np.argsort(indices)])

            # Randomly mix two adjacent horizontal lines.
            indices = np.arange(psds.shape[1]) + np.random.uniform(0, 3, psds.shape[1])
            psds = np.ascontiguousarray(psds[:, np.argsort(indices), :])

            # Random vertical and horizontal flip.
            if np.random.rand() < 0.5:
                psds = np.ascontiguousarray(psds[:, :, ::-1])
            if np.random.rand() < 0.5:
                psds = np.ascontiguousarray(psds[:, ::-1, :])

            # Add random horizontal noise lines.
            if np.random.rand() < 0.1:
                target = np.random.randint(2)
                for _ in range(np.random.choice(3, p=[0.5, 0.3, 0.2])):
                    self._add_horizontal_line(psds[target])

            # Add random horizontal noise beams.
            if np.random.rand() < 0.05:
                target = np.random.randint(2)
                for _ in range(np.random.choice(2, p=[0.7, 0.3])):
                    self._add_horizontal_beam(psds[target])

        return {
            "images": torch.from_numpy(_create_input_image_from_psds(psds)),
            "labels": strength.ceil(),
            "strengths": strength,
        }


@dataclass
class G2NetTestDataset(Dataset):
    filenames: list[str]
    labels: pd.DataFrame | None = None

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        psds = np.load(self.filenames[index]).astype(np.float32)
        output = {"images": torch.from_numpy(_create_input_image_from_psds(psds))}

        if self.labels is not None:
            label = self.labels.loc[os.path.basename(self.filenames[index])[:-4]].target
            output["labels"] = torch.tensor(label, dtype=torch.float32)
        return output


def create_train_val_dataloaders(
    config: DictConfig,
) -> tuple[DataLoader, list[DataLoader]]:
    train_dataloader = DataLoader(
        dataset=G2NetTrainDataset(glob.glob(config.data.train.filenames)),
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        persistent_workers=True,
    )
    val_fake_dataloader = DataLoader(
        dataset=G2NetTrainDataset(
            filenames=glob.glob(config.data.validation_fake.filenames),
            validation=True,
        ),
        batch_size=config.train.batch_size,
        num_workers=os.cpu_count(),
        persistent_workers=True,
    )
    val_real_dataloader = DataLoader(
        dataset=G2NetTestDataset(
            filenames=glob.glob(config.data.validation_real.filenames),
            labels=pd.read_csv(config.data.validation_real.labels, index_col="id"),
        ),
        batch_size=config.train.batch_size,
        num_workers=os.cpu_count(),
        persistent_workers=True,
    )
    return train_dataloader, [val_fake_dataloader, val_real_dataloader]


if __name__ == "__main__":
    import random
    import time

    import matplotlib.pyplot as plt

    filenames = glob.glob("resources/external/train/synthesized/*")
    random.shuffle(filenames)
    dataset = G2NetTrainDataset(filenames)
    # dataset = G2NetPSDDataset(
    #     glob.glob("resources/train/*"),
    #     pd.read_csv("resources/train_labels.csv", index_col="id"),
    # )

    last = time.time()
    """
    for i in range(500):
        x, y = dataset[i]
    print(x.dtype, y.dtype)
    print((time.time() - last) / 500)

    """
    for i in range(10):
        x = dataset[i]
        for j in range(4):
            plt.subplot(1, 4, j + 1)
            plt.title(x["strengths"])
            # plt.imshow(x["images"][j].numpy())
            plt.hist(x["images"][j].flatten(), bins=100)
        plt.show()
