from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import pickle
import random
import shutil

import numpy as np
import pyfstat
import tqdm

logging.getLogger("pyfstat").setLevel(logging.WARNING)


def simulate_random_signal(
    L1_timestamps: np.ndarray,
    H1_timestamps: np.ndarray,
    signal_amplitude: float = 5e-24,
) -> tuple[np.ndarray, np.ndarray]:
    kwargs = {
        "F0": random.uniform(50, 500),
        "F1": random.uniform(-1.01e-9, -1e-9),
        "h0": signal_amplitude,
        **pyfstat.injection_parameters.isotropic_amplitude_distribution,
    }
    kwargs = {
        "outdir": "".join(random.choices("0123456789abcdef", k=16)),
        "detectors": "L1,H1",
        "Tsft": 1800,
        "Band": 0.2,
        "sqrtSX": 0,
        "SFTWindowType": "tukey",
        "SFTWindowBeta": 0.01,
        "timestamps": {"L1": L1_timestamps, "H1": H1_timestamps},
        **pyfstat.AllSkyInjectionParametersGenerator(kwargs).draw(),
    }
    writer = pyfstat.Writer(**kwargs)

    writer.make_data()
    sfts = pyfstat.utils.get_sft_as_arrays(writer.sftfilepath)[2]

    # Remove the temporary directory which is used for storing simulated signals.
    shutil.rmtree(kwargs["outdir"])
    return sfts["L1"][1:], sfts["H1"][1:]


def process_fn(args: argparse.Namespace, index: int, queue: mp.Queue):
    with open(args.timestamps, "rb") as fp:
        timestamps = pickle.load(fp)

    for _ in range(index, args.num_samples, args.cpu_cores):
        L1_SFTs, H1_SFTs = simulate_random_signal(
            *random.choice(timestamps), args.signal_amplitude
        )
        L1_mask = np.abs(L1_SFTs).max(1) > args.signal_threshold
        H1_mask = np.abs(H1_SFTs).max(1) > args.signal_threshold

        # Truncate the available frequency and time range.
        L1_idx = np.arange(L1_SFTs.shape[0])[L1_mask]
        H1_idx = np.arange(H1_SFTs.shape[0])[H1_mask]
        i, j = min(L1_idx.min(), L1_idx.min()), max(H1_idx.max(), H1_idx.max())

        L1_SFTs = L1_SFTs[i:j, : args.time_length]
        H1_SFTs = H1_SFTs[i:j, : args.time_length]

        L1_SFTs = L1_SFTs / (np.abs(L1_SFTs).max(0).mean() / args.normalize_scale)
        H1_SFTs = H1_SFTs / (np.abs(H1_SFTs).max(0).mean() / args.normalize_scale)

        signals = np.stack((L1_SFTs, H1_SFTs), axis=0)
        signals = np.stack((signals.real, signals.imag), axis=0)
        signals = (signals / args.amplitude_scale).astype(np.float16)

        # Save L1 and H1 signals with random file name.
        name = "".join(random.choices("0123456789abcdef", k=16))
        np.save(os.path.join(args.output_directory, f"{name}.npy"), signals)
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
    parser.add_argument("timestamps")
    parser.add_argument("--num-samples", default=30000, type=int)
    parser.add_argument("--signal-amplitude", default=5e-24, type=float)
    parser.add_argument("--signal-threshold", default=1e-22, type=float)
    parser.add_argument("--normalize-scale", default=2e-21, type=float)
    parser.add_argument("--amplitude-scale", default=1e-22, type=float)
    parser.add_argument("--time-length", default=4096, type=int)
    parser.add_argument("--cpu-cores", default=mp.cpu_count(), type=int)
    parser.add_argument("--output-directory", default="resources/external/signals")
    main(parser.parse_args())
