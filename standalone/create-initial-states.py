#!/usr/bin/env python
# coding: utf-8

"""Script to train a SIREN on a given set of images."""

import argparse
from copy import deepcopy
import os
import os.path as osp
import yaml
import torch
from torchvision.transforms.functional import to_pil_image
from ifmorph.dataset import ImageDataset
from ifmorph.model import SIREN


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates the initial states for image morphing, i.e.,"
        " trains a neural network for each input image, so as to represent"
        " them implicitly."
    )
    parser.add_argument("config_path", help="Path to the configuration file.")
    parser.add_argument("images", nargs="+", help="Path to the input images.")
    parser.add_argument(
        "--output-path", default="pretrained",
        help="Folder to store the resulting nets (and optionally, inferences)"
    )
    parser.add_argument(
        "--no-reconstruction", "-r", action="store_true",
        help="Does not perform reconstruction of the images after training."
    )
    parser.add_argument(
        "--device", "-d", type=str, default="cuda:0",
        help="The device to perform the training on."
    )
    parser.add_argument(
        "--nsteps", "-n", type=int, default=0,
        help="Number of training steps for each image."
    )
    parser.add_argument(
        "--batchsize", "-b", type=int, default=0,
        help="Number of pixels to pass to the training loop at each step."
        " Use when training large images. If set to 0 (default), will pass all"
        " points at every step. Any value > 0 will split effectively run"
        " multiple steps, thus \"nsteps\" may be interpreted as \"nepochs\"."
    )
    parser.add_argument(
        "--silent", "-s", action="store_true",
        help="Silences informative outputs. Does not silence errors."
    )
    args = parser.parse_args()

    if not osp.exists(args.config_path):
        raise FileNotFoundError(
            f"Experiment configuration file \"{args.config_path}\" not found."
        )

    with open(args.config_path, "r") as fin:
        config = yaml.safe_load(fin)

    if not osp.exists(args.output_path):
        os.makedirs(args.output_path)

    devstr = args.device
    if "cuda" in devstr and not torch.cuda.is_available():
        devstr = "cpu"
        print("No CUDA available devices found on system. Using CPU.")

    device = torch.device(devstr)

    network_config = config["network"]
    n_channels = network_config["out_channels"]
    model = SIREN(
        n_in_features=network_config["in_channels"],
        n_out_features=n_channels,
        hidden_layer_config=network_config["hidden_layers"],
        w0=network_config["omega_0"],
        ww=network_config.get("omega_w", None),
        delay_init=True
    ).to(device)

    nepochs = args.nsteps
    if not nepochs:
        if "training" not in config:
            nepochs = 1000
        else:
            nepochs = config["training"].get("n_steps", 1000)

    for fname in args.images:
        img = ImageDataset(
            fname, channels_to_use=n_channels, batch_size=args.batchsize
        )
        steps_per_epoch = len(img)

        if not args.silent:
            print(f"Training for image: \"{fname}\" -- Epochs: {nepochs}"
                  f" -- Batch size: {img.batch_size}"
                  f" -- Steps per epoch {steps_per_epoch}")

        model.update_omegas(
            w0=network_config["omega_0"],
            ww=network_config.get("omega_w", model.w0)
        )
        model.reset_weights()
        model.train()
        optim = torch.optim.Adam(
            lr=config["optimizer"].get("lr", 1e-4),
            params=model.parameters()
        )

        best_loss = torch.inf
        best_weights = {}
        best_epoch = 0
        for epoch in range(nepochs):
            epochloss = 0
            for step in range(steps_per_epoch):
                X, y, idx = img.__getitem__()
                X = X.detach().to(device).requires_grad_(False)
                y = y.detach().to(device).requires_grad_(False)
                yhat = model(X, preserve_grad=True)["model_out"]
                loss = torch.pow(y - yhat, 2).sum(1).mean()

                epochloss += loss.item()
                optim.zero_grad()
                loss.backward()
                optim.step()

            if not args.silent and epoch and not epoch % (nepochs // 20):
                print(f"Epoch: {epoch} -- Loss: {epochloss}")

            if epochloss < best_loss and epoch > (nepochs // 2):
                best_epoch = epoch
                best_loss = epochloss
                best_weights = deepcopy(model.state_dict())

        if not args.silent:
            print(f"Best loss at epoch {best_epoch} = {best_loss}")

        model.load_state_dict(best_weights)
        model.update_omegas()
        out_basename = osp.split(fname)[1].split(".")[0]
        torch.save(
            model.state_dict(),
            osp.join(args.output_path, out_basename + ".pth")
        )

        if not args.no_reconstruction:
            model.eval()
            with torch.no_grad():
                idx = torch.arange(img.rgb.shape[0], device=device)
                rec = torch.zeros_like(img.rgb, device=device).requires_grad_(False)
                for step in range(steps_per_epoch):
                    idxslice = idx[step * img.batch_size:(step + 1) * img.batch_size]
                    X, _, _ = img.__getitem__(idxslice)
                    X = X.detach().to(device).requires_grad_(False)
                    yhat = model(X, preserve_grad=True)["model_out"]
                    rec[idxslice, ...] = yhat

                rec = rec.detach().cpu().clip(0, 1).requires_grad_(False)

            sz = [img.size[0], img.size[1], n_channels]
            rec = rec.reshape(sz).permute((2, 0, 1))
            img = to_pil_image(rec)
            out_img_path = osp.join(args.output_path, out_basename + ".png")
            img.save(out_img_path)
            print(f"Image saved as: {out_img_path}")

    print("Done")
