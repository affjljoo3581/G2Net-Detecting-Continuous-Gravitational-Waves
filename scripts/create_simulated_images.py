from __future__ import annotations

import argparse
import glob
import os

import h5py
import lalpulsar
import numpy as np
import pyfstat
import tqdm

from utils import create_input_image_from_SFTs


def create_simulated_SFTs(
    L1_timestamps: np.ndarray, H1_timestamps: np.ndarray, signal: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    ...


def main(args: argparse.Namespace):
    os.makedirs(args.output_directory, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("timestamps")
    parser.add_argument("--num-samples", default=100000, type=int)
    parser.add_argument("--max-length", default=4096, type=int)
    parser.add_argument("--stride", default=32, type=int)
    parser.add_argument("--scale", default=1e22, type=float)
    parser.add_argument("--output-directory", default="resources/train")
    main(parser.parse_args())
