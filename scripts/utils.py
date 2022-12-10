from __future__ import annotations

import numpy as np


def _normalize(
    array: np.ndarray, axis: int | tuple[int, ...] | None = None
) -> np.ndarray:
    minimum, maximum = array.min(axis, keepdims=True), array.max(axis, keepdims=True)
    return (array - minimum) / (maximum - minimum)


def create_input_image_from_SFTs(
    L1_SFTs: np.ndarray,
    H1_SFTs: np.ndarray,
    max_length: int = 4096,
    stride: int = 32,
    scale: float = 1e22,
) -> np.ndarray:
    # Get scaled power spectrums of L1 and H1 from their SFTs.
    L1_spec, H1_spec = (np.abs(L1_SFTs) * scale) ** 2, (np.abs(H1_SFTs) * scale) ** 2

    # Remove vertical noises and normalize to be in the range of [0, 1].
    L1_spec = L1_spec - L1_spec.mean(0, keepdims=True)
    H1_spec = H1_spec - H1_spec.mean(0, keepdims=True)
    L1_spec, H1_spec = _normalize(L1_spec), _normalize(H1_spec)

    # Truncate the spectrum and slice into chunks.
    L1_spec = L1_spec[:, :max_length].reshape(-1, max_length // stride, stride)
    H1_spec = H1_spec[:, :max_length].reshape(-1, max_length // stride, stride)

    L1_mean, H1_mean = L1_spec.mean(2), H1_spec.mean(2)
    mbm_L1_H1, mam_L1_H1 = (L1_spec * H1_spec).mean(2), L1_mean * H1_mean
    mam_exp_L1_H1 = np.exp(mam_L1_H1**2)

    image = np.stack((L1_mean, H1_mean, mbm_L1_H1, mam_L1_H1, mam_exp_L1_H1), axis=2)
    return _normalize(image, axis=(0, 1))
