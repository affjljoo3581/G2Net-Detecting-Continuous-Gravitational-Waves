from __future__ import annotations

import argparse
import glob
import os
import pickle

import h5py
import tqdm


def main(args: argparse.Namespace):
    timestamps = []
    for filename in tqdm.tqdm(glob.glob(os.path.join(args.directory, "*.hdf5"))):
        with h5py.File(filename) as data:
            data = data[list(data.keys())[0]]
            L1_timestamps = data["L1/timestamps_GPS"][:]
            H1_timestamps = data["H1/timestamps_GPS"][:]
            timestamps.append((L1_timestamps, H1_timestamps))

    with open(args.output_filename, "wb") as fp:
        pickle.dump(timestamps, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--output-filename", default="resources/timestamps.pkl")
    main(parser.parse_args())
