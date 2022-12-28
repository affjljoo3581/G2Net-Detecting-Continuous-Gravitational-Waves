from __future__ import annotations

import argparse
import glob
import os

import h5py
import numpy as np
import tqdm


def main(args: argparse.Namespace):
    os.makedirs(args.output_directory, exist_ok=True)

    for filename in tqdm.tqdm(glob.glob(os.path.join(args.directory, "*.hdf5"))):
        with h5py.File(filename) as data:
            name = list(data.keys())[0]
            L1_SFTs = np.array(data[f"{name}/L1/SFTs"], dtype=np.complex128)
            H1_SFTs = np.array(data[f"{name}/H1/SFTs"], dtype=np.complex128)

            # Skip invalid SFTs which are for easter-eggs or too short to aggregate.
            if (
                L1_SFTs.shape[0] != args.freq_length
                or H1_SFTs.shape[0] != args.freq_length
                or L1_SFTs.shape[1] < args.time_length
                or H1_SFTs.shape[1] < args.time_length
            ):
                continue

            # Calculate the power spectral density.
            L1_psd = np.abs(L1_SFTs / args.amplitude_scale) ** 2
            H1_psd = np.abs(H1_SFTs / args.amplitude_scale) ** 2

            # Truncate the time range and average by splitting into chunks.
            L1_psd = L1_psd[:, : args.time_length]
            H1_psd = H1_psd[:, : args.time_length]

            L1_psd = L1_psd.reshape(L1_psd.shape[0], -1, args.window_size)
            H1_psd = H1_psd.reshape(H1_psd.shape[0], -1, args.window_size)
            L1_mean, H1_mean = L1_psd.mean(2), H1_psd.mean(2)

            compressed = np.stack((L1_mean, H1_mean), axis=0)
            np.save(os.path.join(args.output_directory, f"{name}.npy"), compressed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--amplitude-scale", default=1e-22, type=int)
    parser.add_argument("--freq-length", default=360, type=int)
    parser.add_argument("--time-length", default=4096, type=int)
    parser.add_argument("--window-size", default=32, type=int)
    parser.add_argument("--output-directory", default="resources/competition/test")
    main(parser.parse_args())
