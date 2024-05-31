#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import os.path as osp
import sys
import cv2
import numpy as np
import pandas as pd
import lpips
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculates LPIPS score between original and morphed"
        " images."
    )
    parser.add_argument(
        "originaldir", help="Path to the original images used for morphing."
    )
    parser.add_argument(
        "blendeddir", help="Path to the morphed images. Note that we expect the"
        " filenames to be P1-P2.$\{EXT\}, where P1 and P2 are the original"
        " images used for morphing."
    )
    parser.add_argument(
        "--morphingpairs", "-p",
        default=osp.join("data", "pairs_for_morphing_full.txt"), help="Path to"
        " a text file with the morphing pairs."
        " See \"data/pairs_for_morphing_full.txt\" (Default value) for an"
        " example."
    )
    args = parser.parse_args()

    loss_fn = lpips.LPIPS(net="alex")

    with open(args.morphingpairs, 'r') as fin:
        pairs = [ln.strip().split() for ln in fin.readlines()]
        for i, p in enumerate(pairs):
            p[0] = p[0][:3]
            p[1] = p[1][:3]
            pairs[i] = p

    BASEIMDIR = args.originaldir
    MORPHINGDIR = args.blendeddir

    baseims = dict()
    for fname in os.listdir(BASEIMDIR):
        impath = osp.join(BASEIMDIR, fname)
        im = cv2.cvtColor(cv2.imread(impath), cv2.COLOR_BGR2RGB).astype(np.float32)
        im = (im - 128.0) / 128.0
        im = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0)
        baseims[osp.splitext(fname)[0]] = im

    stats = dict()
    with torch.no_grad():
        for p in pairs:
            morphfname = f"{p[0]}-{p[1]}.png"
            impath = osp.join(MORPHINGDIR, morphfname)
            if not osp.exists(impath):
                print(
                    f"File \"{impath}\" does not exist. Is the extension"
                     " \"png\"?",
                    file=sys.stderr
                )
                continue

            morphim = (cv2.cvtColor(
                cv2.imread(impath), cv2.COLOR_BGR2RGB
            ).astype(np.float32) - 128.) / 128.
            morphim = torch.from_numpy(morphim).permute(2, 0, 1).unsqueeze(0)

            dsrc = loss_fn(baseims[p[0]], morphim)
            dtgt = loss_fn(baseims[p[1]], morphim)
            stats[f"{p[0]}-{p[1]}"] = [dsrc.item(), dtgt.item()]

        df = pd.DataFrame.from_dict(stats).T
