#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import os.path as osp
import sys
import numpy as np
import yaml


DEFAULT_EXP_CONFIG = {
    "description": "warping with feature matching pairwise morphs",
    "device": "cuda:0",
    "experiment_name": "",
    "initial_conditions": {},
    "loss": {
        "sources": [],
        "targets": [],
        "constraint_weights": {
            "data_constraint": 1e4,
            "identity_constraint": 1e3,
            "inv_constraint": 1e4,
            "TPS_constraint": 1e3,
        },
        "intermediate_times": [0.16, 0.32, 0.5, 0.66, 0.82],
        "type": "featurematching"
    },
    "network": {
        "hidden_layers": [128, 128],
        "in_channels": 3,
        "out_channels": 2,
        "omega_0": 24,
        "omega_w": 24,
    },
    "optimizer": {
        "lr": 0.0001,
    },
    "reconstruct": {
        "fps": 20,
        "frame_dims": [512, 512],
        "n_frames": 101,
        "timerange": [-1.0, 1.0],
    },
    "training": {
        # "checkpoint_steps": 5000,
        "n_samples": 20000,
        "n_steps": 10001,
        # "reconstruction_steps": 10000,
        "warmup_steps": 1000,
    }
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch creation of experiment files given a list of"
        " morphing pairs."
    )
    parser.add_argument(
        "pairlist", help="File containing the list of pairs for morphing. See"
        " the example in \"data/pairs_for_morphing_full.txt\"."
    )
    parser.add_argument(
        "imagedir", help="Path to the input images folder. Note that the"
        " images may be encoded as neural networks or normal images."
        " If encoded as neural networks, we expect their filenames to end in "
        " \".pt{h}\"."
    )
    parser.add_argument(
        "output_path", help="Path to the output experiment files. All saved in"
        " YAML format."
    )
    parser.add_argument(
        "--lmdir", default="", type=str, help="Path to the landmark file"
        " directory. Optional, by default we search the image directory for"
        " the landmark files. Note that we assume the landmark files to have"
        " a \".dat\" extension."
    )
    args = parser.parse_args()

    with open(args.pairlist, 'r') as fin:
        contents = [line.strip() for line in fin.readlines()]

    unique_is = set()
    pairs_morphing = dict()
    for line in contents:
        f1, f2 = line.split(' ')
        f1 = osp.splitext(f1)[0]
        f2 = osp.splitext(f2)[0]
        if f1 == f2:
            print(
                f"[WARNING] Asking to morph \"{f1}\" and \"{f2}\" when they"
                " are the same image. Skipping this pair.", file=sys.stderr)
            continue

        unique_is.add(f1)
        unique_is.add(f2)
        if f1 not in pairs_morphing:
            pairs_morphing[f1] = [f2]
        else:
            pairs_morphing[f1].append(f2)

    if not osp.exists(args.output_path):
        os.makedirs(args.output_path)

    LMDIR = args.lmdir if args.lmdir else args.imagedir
    imagenames = dict()
    for f in os.listdir(args.imagedir):
        if "DS_Store" in f:
            continue

        fname = osp.join(args.imagedir, f)
        if osp.isdir(fname):
            continue

        imagenames[osp.splitext(f)[0]] = osp.join(args.imagedir, f)

    lmfnames = [f for f in os.listdir(LMDIR) if f.endswith(".dat")]
    # Pre-loading landmarks once to avoid overloading the disk
    lmdict = {}
    for fname in lmfnames:
        lmdict[osp.splitext(fname)[0]] = np.load(
            osp.join(LMDIR, fname), allow_pickle=True
        )

    for src, tgtlist in pairs_morphing.items():
        for tgt in tgtlist:
            config = DEFAULT_EXP_CONFIG
            config["initial_conditions"] = {
                "0": osp.join(imagenames[src]),
                "1": osp.join(imagenames[tgt]),
            }
            try:
                config["loss"]["sources"] = lmdict[src].tolist()
            except KeyError:
                print(f"No landmarks for \"{src}\". Skipping.")
                continue
            try:
                config["loss"]["targets"] = lmdict[tgt].tolist()
            except KeyError:
                print(f"No landmarks for \"{tgt}\". Skipping.")
                continue

            with open(osp.join(args.output_path, f"{src}-{tgt}.yaml"), "w+") as fout:
                yaml.dump(config, fout)
