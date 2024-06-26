#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import os.path as osp
import sys
import cv2
import torch
from ifmorph.model import from_pth
from ifmorph.util import get_landmarks_dlib, image_inference


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detects landmarks using DLib from a series of images"
        " encoded as neural networks."
    )
    parser.add_argument(
        "models", nargs="+", help="Path to the PyTorch files representing"
        " face images."
    )
    parser.add_argument(
        "--outputdir", "-o", default=".", help="Path to the output directory"
        " where the landmark files will be stored. It will be created if it"
        " does not exists."
    )
    parser.add_argument(
        "--device", "-d", default="cuda:0", help="Device to run the inference."
        " By default its \"cuda:0\". Will switch to \"cpu\" if no CUDA capable"
        " device is found."
    )
    parser.add_argument(
        "--dims", type=str, default="512,512", help="Image dimensions to"
        " consider when running the inference. We use Width,Height notation."
        " Note that DLib runs on a face image, thus larger images may result"
        " in more precise landmark placement. Default is 512,512."
    )
    parser.add_argument(
        "--saveim", "-s", action="store_true", default=False, help="Saves the"
        " output images."
    )
    parser.add_argument(
        "--plot-landmarks", "-p", action="store_true", default=False,
        help="Plots the landmarks over the saved images. Only makes sense"
        " when passing \"--saveim\" as well, ignored otherwise."
    )
    args = parser.parse_args()

    devstr = args.device
    if "cuda" in args.device and not torch.cuda.is_available():
        print("No CUDA-capable device found. Switching to CPU.",
              file=sys.stderr)
        devstr = "cpu"

    device = torch.device(devstr)
    dims = [int(d) for d in args.dims.split(",")]

    if not osp.exists(args.outputdir):
        os.makedirs(args.outputdir)

    lmdict = {}
    for modelname in args.models:
        faceim = from_pth(modelname, device=device).eval()
        img = image_inference(faceim, dims, device=device)
        lms = get_landmarks_dlib(img).astype(float)
        key = osp.splitext(osp.split(modelname)[1])[0]

        if args.saveim:
            imname = osp.splitext(osp.split(modelname)[-1])[0] + ".png"
            if args.plot_landmarks:
                img[lms[:, 0], lms[:, 1], :] = (0, 255, 0)
            cv2.imwrite(
                osp.join(args.outputdir, imname),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            )

        lms[:, 0] = 2.0 * (lms[:, 0] / img.shape[0]) - 1.0
        lms[:, 1] = 2.0 * (lms[:, 1] / img.shape[1]) - 1.0
        lmdict[key] = lms

    for lmname, lm in lmdict.items():
        lm.dump(osp.join(args.outputdir, lmname) + ".dat")
