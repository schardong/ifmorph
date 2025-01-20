#!/usr/bin/env python
# coding: utf-8

"""Runs the inference on a pre-trained Flow model. Note that the output is a
video. See `warp-inference-images.py` for the image output script."""

import argparse
import os
import os.path as osp
import torch
import yaml
from ifmorph.dataset import check_network_type, ImageDataset, NotTorchFile
from ifmorph.model import from_pth
from ifmorph.util import create_morphing

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
        "--output-path", "-o", default=os.getcwd(),
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
        "--framedims", "-f", nargs='+', help="Dimensions (in pixels) for the"
        " output image. Note that it must contain two numbers separated by a"
        " space, e.g. \"-f 800 600\"."
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

    reconstruct_config = config["reconstruct"]
    if args.framedims:
        grid_dims = [int(d) for d in args.framedims]
    else:
        grid_dims = reconstruct_config.get("frame_dims", [640, 640])

    initialstates = [None] * len(config["initial_conditions"])
    for i, p in enumerate(config["initial_conditions"].values()):
        try:
            nettype = check_network_type(p)
        except NotTorchFile:
            initialstates[i] = ImageDataset(p, sidelen=grid_dims)
        else:
            if nettype == "siren":
                initialstates[i] = from_pth(p, w0=1, device=device)
            elif nettype == "mrnet" and WITH_MRNET:
                initialstates[i] = MRFactory.load_state_dict(p).to(device)
            else:
                raise ValueError(f"Unknown network type: {nettype}")

    vidfilename = f"video_{args.checkpoint}_{args.blending}"
    if args.landmarks:
        vidfilename += "_landmarks"
    vidfilename += ".mp4"
    if not args.output_path:
        vidpath = osp.join(basepath, vidfilename)
    else:
        vidpath = osp.join(args.output_path, vidfilename)

    timerange = reconstruct_config.get("timerange", [-1, 1])
    n_frames = reconstruct_config.get("n_frames", 100)
    fps = reconstruct_config.get("fps", 10)

    morph_sources = torch.Tensor(config["loss"]["sources"]).float().to(device)
    morph_targets = torch.Tensor(config["loss"]["targets"]).float().to(device)
    create_morphing(
        warp_net=model,
        frame0=initialstates[0],
        frame1=initialstates[1],
        output_path=vidpath,
        frame_dims=grid_dims,
        n_frames=n_frames,
        fps=fps,
        device=device,
        landmark_src=morph_sources,
        landmark_tgt=morph_targets,
        overlay_landmarks=args.landmarks,
        blending_type=args.blending,
    )

    print(f"Output video written to \"{vidpath}\"")
