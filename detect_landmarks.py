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
        "inputdir", help="Path to the PTH files representing the face images."
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
        help="Plots the landamarks over the saved images. Only makes sense"
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

    modelnames = [osp.join(args.inputdir, f) for f in os.listdir(args.inputdir) if f.endswith(".pth")]
    lmdict = {}
    for modelname in sorted(modelnames):
        faceim = from_pth(modelname, device=device).eval()
        img = image_inference(faceim, dims, device=device)
        lms = get_landmarks_dlib(img)
        key = osp.splitext(osp.split(modelname)[1])[0]
        lmdict[key] = lms

        if args.saveim:
            if args.plot_landmarks:
                img[lms[:, 0], lms[:, 1], :] = (0, 255, 0)
            cv2.imwrite(
                modelname[:-3] + "png",
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            )

    for lmname, lm in lmdict.items():
        lm.dump(osp.join(args.inputdir, lmname) + ".dat")
