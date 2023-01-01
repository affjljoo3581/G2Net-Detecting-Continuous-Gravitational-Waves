from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import random

import cv2
import numpy as np
import torch
import tqdm


def augment_signals(signals: torch.Tensor) -> torch.Tensor:
    if np.random.rand() < 0.5:
        signals = np.ascontiguousarray(signals[:, :, :, ::-1])
    if np.random.rand() < 0.5:
        signals = np.ascontiguousarray(signals[:, :, ::-1, :])
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.8, 1.2)
        signals = signals.reshape(-1, *signals.shape[2:]).transpose(1, 2, 0)
        signals = cv2.resize(signals, None, fx=1.0, fy=scale)
        signals = signals.transpose(2, 0, 1).reshape(2, 2, *signals.shape[:2])
    return signals


def create_synthesized_psds(
    args: argparse.Namespace, signals: torch.Tensor | None = None, strength: float = 0.0
) -> torch.Tensor:
    noises = torch.normal(0, args.noise_std, (2, 2, args.freq_length, args.time_length))
    noises = noises.numpy()

    if signals is not None:
        signals = strength * augment_signals(signals)
        offset = np.random.randint(args.freq_length - signals.shape[2])
        noises[:, :, offset : offset + signals.shape[2]] += signals

    psds = np.square(noises)
    psds = psds.reshape(2, 2, args.freq_length, -1, args.window_size).mean(4).sum(0)
    return psds


def process_fn(args: argparse.Namespace, index: int, queue: mp.Queue):
    filenames = glob.glob(os.path.join(args.directory, "*.npy"))

    for _ in range(index, args.num_samples, args.cpu_cores):
        name = "".join(random.choices("0123456789abcdef", k=16))

        signals, strength = None, 0
        if np.random.rand() < 0.5:
            signals = np.load(np.random.choice(filenames)).astype(np.float32)
            strength = np.random.uniform(args.low_strength, args.high_strength)

        psds = create_synthesized_psds(args, signals, strength)
        filename = f"{name}-{int(strength * 1e8):08d}.npy"

        np.save(os.path.join(args.output_directory, filename), psds.astype(np.float16))
        queue.put(None)


def main(args: argparse.Namespace):
    os.makedirs(args.output_directory, exist_ok=True)

    processes, queue = [], mp.Queue()
    for i in range(args.cpu_cores):
        p = mp.Process(target=process_fn, args=(args, i, queue), daemon=True)
        processes.append(p)
        p.start()

    for _ in tqdm.trange(args.num_samples):
        queue.get()
    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--num-samples", default=1000000, type=int)
    parser.add_argument("--noise-std", default=1.0587912, type=float)
    parser.add_argument("--freq-length", default=360, type=int)
    parser.add_argument("--time-length", default=4096, type=int)
    parser.add_argument("--window-size", default=32, type=int)
    parser.add_argument("--low-strength", default=0.01, type=float)
    parser.add_argument("--high-strength", default=0.05, type=float)
    parser.add_argument("--cpu-cores", default=mp.cpu_count(), type=int)
    parser.add_argument("--output-directory", default="resources/external/synthesized")
    main(parser.parse_args())
