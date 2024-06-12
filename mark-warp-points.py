#!/usr/bin/env python
# coding: utf-8

"""Script to mark the warping points on two images represented by neural
networks. Optionally, may use automatic methods to provide a first guess of the
important feature points."""


import argparse
import os.path as osp
import sys
import cv2
import torch
import yaml
from ifmorph.model import from_pth
from ifmorph.util import get_landmark_correspondences


DEFAULT_WARPING_CONFIG = {
    "description": "warping_with_feature_matching",
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
        "checkpoint_steps": 5000,
        "n_samples": 20000,
        "n_steps": 21001,
        "reconstruction_steps": 10000,
        "warmup_steps": 1000,
    }
}
FRAME_DIMS = (600, 600)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Marking points for warping. Outputs a skeleton experiment"
        " configuration file."
    )
    parser.add_argument(
        "source", help="The warping source image (encoded as a neural"
        " network)."
    )
    parser.add_argument(
        "target", help="The warping target image (also encoded as a neural"
        " network)."
    )
    parser.add_argument(
        "--source_landmarks", "-s", default="",
        help="TEM formatted file with the source landmarks. See the FRLL"
        " dataset information"
    )
    parser.add_argument(
        "--target_landmarks", "-t", default="",
        help="TEM formatted file with the target landmarks. See the FRLL"
        " dataset information"
    )
    parser.add_argument(
        "--template", default="", help="Use this preconfigured YAML file"
        " as a template. Replacing the initial conditions and warping sources"
        " and targets."
    )
    parser.add_argument(
        "--output", "-o", default="output.yaml", help="The output path to the"
        " experiment file."
    )
    parser.add_argument(
        "--landmark_detector", "-l", default="",
        help="Landmark detector to use. Options are \"none\" (or empty),"
        " \"dlib\" and \"mediapipe\". Leave as is for no landmark detection."
        " Start in a clean slate (or from template)."
    )
    parser.add_argument(
        "--device", "-d", default="cuda:0", help="Device to run the inference."
        " By default its \"cuda:0\". Will switch to \"cpu\" if no CUDA capable"
        " device is found."
    )
    parser.add_argument(
        "--no_ui", "-n", action="store_true", help="Don't open the point"
        " editing UI. Useful for batch runs. Note that for this option to be"
        " useful, \"--landmark_detector\" must not be empty."
    )
    args = parser.parse_args()

    if args.no_ui and not (args.landmark_detector or args.template):
        print("[WARNING] Set to not open UI, nor automatically detect landmark"
              " of fetch them from previous file. Check the output file"
              f" \"{args.output}\" to see if there are any landmarks.")

    device = torch.device(args.device)
    if "cuda" in args.device and not torch.cuda.is_available():
        print("No CUDA-capable device found. Switching to CPU.")
        device = torch.device("cpu")

    pathsrc = args.source
    pathtgt = args.target

    input_experiment_file = args.template
    output_experiment_file = args.output

    config = None
    if osp.exists(input_experiment_file):
        with open(input_experiment_file, 'r') as fin:
            config = yaml.safe_load(fin)

    try:
        frame0 = from_pth(pathsrc, device=device)
    except Exception:
        frame0 = cv2.cvtColor(cv2.imread(pathsrc), cv2.COLOR_BGR2RGB)

    try:
        frame1 = from_pth(pathtgt, device=device)
    except Exception:
        frame1 = cv2.cvtColor(cv2.imread(pathtgt), cv2.COLOR_BGR2RGB)

    src = []
    tgt = []
    if config is not None and "loss" in config:
        src = torch.tensor(
            config["loss"].get("sources", []), device=device
        ).requires_grad_(False)
        tgt = torch.tensor(
            config["loss"].get("targets", []), device=device
        ).requires_grad_(False)

    if args.source_landmarks:
        lm = []
        with open(args.source_landmarks, 'r') as fin:
            lm = yaml.safe_load(fin)

        lmt = torch.Tensor(lm).to(device).requires_grad_(False)
        if src:
            src = torch.cat((src, lmt), dim=0)
        else:
            src = lmt.clone()

    if args.target_landmarks:
        lm = []
        with open(args.target_landmarks, 'r') as fin:
            lm = yaml.safe_load(fin)

        lmt = torch.Tensor(lm).to(device).requires_grad_(False)
        if tgt:
            tgt = torch.cat((tgt, lmt), dim=0)
        else:
            tgt = lmt.clone()

    try:
        src, tgt = get_landmark_correspondences(
            frame0, frame1, FRAME_DIMS, src_pts=src, tgt_pts=tgt,
            device=device, method=args.landmark_detector,
            open_ui=not args.no_ui
        )
    except ValueError:
        print(f"Unrecognized option: \"{args.landmark_detector}\".",
              file=sys.stderr)

    if config is None:
        config = DEFAULT_WARPING_CONFIG

    config["initial_conditions"] = {
        "0": pathsrc,
        "1": pathtgt
    }

    if src is not None and len(src):
        config["loss"]["sources"] = src.tolist()

    if tgt is not None and len(tgt):
        config["loss"]["targets"] = tgt.tolist()

    with open(output_experiment_file, 'w') as fout:
        yaml.dump(config, fout)
