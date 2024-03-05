#!/usr/bin/env python
# coding: utf-8

"""Runs the inference on a pre-trained Flow model. Note that the output is a
video. See `warp-inference-images.py` for the image output script."""

import argparse
import os
import os.path as osp
import numpy as np
import torch
import yaml
from ifmorph.dataset import check_network_type
from ifmorph.model import from_pth
from ifmorph.util import create_morphing_video

WITH_MRNET = True
try:
    from ext.mrimg.src.networks.mrnet import MRFactory
except (ModuleNotFoundError, ImportError):
    WITH_MRNET = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "configpath",
        help="Path to experiment configuration file stored with the output"
        " PTHs."
    )
    parser.add_argument(
        "--outputdir", "-o", default=os.getcwd(),
        help="Path to the output file. By default is the current working"
        "directory , and the file is named after the chosen checkpoint, or"
        " \"video.mp4\" for the final checkpoint."
    )
    parser.add_argument(
        "--landmarks", "-l", default=False, action="store_true",
        help="Whether to overlay the source/target landmarks on the resulting"
        " video."
    )
    parser.add_argument(
        "--rots", default=0, type=int, help="Rotation angles (in degrees) to"
        " apply to the source image."
    )
    parser.add_argument(
        "--rott", default=0, type=int, help="Rotation angles (in degrees) to"
        " apply to the target image."
    )
    parser.add_argument(
        "--tsx", default=0, type=float, help="Translation of X coordinate in"
        " the source image."
    )
    parser.add_argument(
        "--tsy", default=0, type=float, help="Translation of F coordinate in"
        " the source image."
    )
    parser.add_argument(
        "--ttx", default=0, type=float, help="Translation of X coordinate in"
        " the target image."
    )
    parser.add_argument(
        "--tty", default=0, type=float, help="Translation of Y coordinate in"
        " the target image."
    )
    parser.add_argument(
        "--device", "-d", default="cuda:0",
        help="The device to run the inference on. By default its set as"
        " \"cuda:0\" If CUDA is not supported, then the CPU will be used."
    )
    parser.add_argument(
        "--checkpoint", "-c", default="best",
        help="The checkpoint weigths to perform reconstruction. By default"
        " we use the best weights, saved as \"weights.pth\". Note that this is"
        " a number which will be used to compose the name"
        " \"checkpoint_CHECKPOINT.pth\", unless the default value is kept."
    )
    parser.add_argument(
        "--blending", "-b", default="linear", type=str,
        help="The type of blending to use. May be any of \"linear\", \"min\","
        " \"max\", \"dist\", \"src\", \"tgt\", \"seamless_{clone,mix}\"."
    )
    args = parser.parse_args()

    devstr = args.device
    if "cuda" in devstr and not torch.cuda.is_available():
        devstr = "cpu"
        print("No CUDA available devices found on system. Using CPU.")
    else:
        torch.cuda.empty_cache()
    device = torch.device(devstr)

    if not osp.exists(args.configpath):
        raise FileNotFoundError("Configuration file not found at"
                                f" \"{args.configfile}\". Aborting.")

    with open(args.configpath, 'r') as fin:
        config = yaml.safe_load(fin)

    network_config = config["network"]

    modelfilename = "weights.pth"
    warping_omega0 = 1
    warping_omegaW = 1
    if args.checkpoint != "best":
        modelfilename = f"checkpoint_{args.checkpoint}.pth"
        warping_omega0 = network_config["omega_0"]
        warping_omegaW = network_config["omega_w"]

    basepath = osp.split(osp.expanduser(args.configpath))[0]
    modelpath = osp.join(basepath, modelfilename)
    if not osp.exists(modelpath):
        raise FileNotFoundError(f"Model file \"{modelpath}\" not found.")

    model = from_pth(modelpath, w0=warping_omega0, ww=warping_omegaW,
                     device=device)

    shapenets = [None] * len(config["initial_conditions"])
    for i, p in enumerate(config["initial_conditions"].values()):
        nettype = check_network_type(p)
        if nettype == "siren":
            shapenets[i] = from_pth(p, w0=1, device=device)
        elif nettype == "mrnet" and WITH_MRNET:
            shapenets[i] = MRFactory.load_state_dict(p).to(device)
        else:
            raise ValueError(f"Unknown network type: {nettype}")

    vidfilename = f"video_{args.checkpoint}_{args.blending}"
    if args.landmarks:
        vidfilename += "_landmarks"
    vidfilename += ".mp4"
    if not args.outputdir:
        vidpath = osp.join(basepath, vidfilename)
    else:
        vidpath = osp.join(args.outputdir, vidfilename)

    reconstruct_config = config["reconstruct"]
    timerange = reconstruct_config.get("timerange", [-1, 1])
    n_frames = reconstruct_config.get("n_frames", 100)
    fps = reconstruct_config.get("fps", 10)
    grid_dims = reconstruct_config.get("frame_dims", [320, 320])

    morph_sources = torch.tensor(
        config["loss"]["sources"], dtype=torch.float32, device=device
    )
    morph_targets = torch.tensor(
        config["loss"]["targets"], dtype=torch.float32, device=device
    )

    create_morphing_video(
        warp_net=model,
        shape_net0=shapenets[0],
        shape_net1=shapenets[1],
        output_path=vidpath,
        frame_dims=grid_dims,
        n_frames=n_frames,
        fps=fps,
        ic_times=timerange,
        device=device,
        src=morph_sources,
        tgt=morph_targets,
        plot_landmarks=args.landmarks,
        blending_type=args.blending,
        angles=(np.deg2rad(args.rots), np.deg2rad(args.rott)),
        translations=[[args.tsx, args.tsy], [args.ttx, args.tty]]
    )

    print(f"Output video written to \"{vidpath}\"")
