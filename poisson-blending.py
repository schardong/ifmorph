#!/usr/bin/env python
# coding: utf-8

import argparse
from enum import Enum
import math
import os
import os.path as osp
import random
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from ifmorph.diff_operators import jacobian
from ifmorph.model import from_pth, SIREN
from ifmorph.point_editor import FaceInteractor
from ifmorph.util import get_grid, get_silhouette_lm, get_bottom_face_lm


DEFAULT_BLENDING_WEIGHTS = {
    "grad_constraint": 1,
    "pixel_constraint": 6e3,
}


def gradients_mse(pred_image, coords, gt_jac):
    # compute jacobian on the model
    jac = jacobian(pred_image, coords)[0]
    # compare them with the ground-truth
    return torch.mean((jac - gt_jac[..., :2])**2).sum(-1)


def pixels_mse(pred_image, gt_image):
    return torch.mean((pred_image - gt_image)**2)


class GradientBlendingLoss(torch.nn.Module):
    def __init__(self, constraint_weights=DEFAULT_BLENDING_WEIGHTS):
        super(GradientBlendingLoss, self).__init__()
        self.constraint_weights = constraint_weights

    def forward(
            self, in_mask_output, out_mask_output, gt_jacobian, gt_img
    ):
        grad_loss = gradients_mse(
            in_mask_output["model_out"], in_mask_output["model_in"],
            gt_jacobian
        )
        pixel_loss = pixels_mse(
            out_mask_output["model_out"], gt_img)

        return {
            "grad_constraint": grad_loss * self.constraint_weights["grad_constraint"],
            "pixel_constraint": pixel_loss * self.constraint_weights["pixel_constraint"]
        }


class PoissonEqn(Dataset):
    def __init__(self, values, jac, coords, device=torch.device("cpu")):
        # Compute gradient and laplacian
        self.coords = coords.to(device)
        self.values = values.to(device)
        self.jac = jac.to(device)

    def __len__(self):
        return 1

    def __getitem__(self, _):
        return self.coords, {"jac": self.jac, "values": self.values}


def slerp(val, low, high):
    low_norm = low/(torch.norm(low, dim=1, keepdim=True))
    high_norm = high/(torch.norm(high, dim=1, keepdim=True))
    prod = (low_norm*high_norm).sum(1).unsqueeze(-1)
    omega = torch.where(
        abs(prod) > 0.9,
        torch.zeros_like(prod) + 1e-7,
        torch.acos(prod.clamp(-0.999999, 0.999999))
    )
    # omega = torch.acos(
    #     (low_norm*high_norm).sum(1).clamp(-0.999999, 0.999999)
    # ).unsqueeze(-1)
    so = torch.sin(omega)
    # torch.where(so!=0)
    res = (torch.sin((1.0-val)*omega)/so) * low +\
        (torch.sin(val*omega)/so) * high
    return res


def slerp_Jacobian(val, U, V):
    U_copy = U.clone()
    V_copy = V.clone()

    U[..., 0, :] /= torch.norm(U[..., 0, :], dim=1, keepdim=True)
    U[..., 1, :] /= torch.norm(U[..., 1, :], dim=1, keepdim=True)
    U[..., 2, :] /= torch.norm(U[..., 2, :], dim=1, keepdim=True)

    V[..., 0, :] /= torch.norm(V[..., 0, :], dim=1, keepdim=True)
    V[..., 1, :] /= torch.norm(V[..., 1, :], dim=1, keepdim=True)
    V[..., 2, :] /= torch.norm(V[..., 2, :], dim=1, keepdim=True)

    prod0 = (U[..., 0, :]*V[..., 0, :]).sum(1).unsqueeze(-1)
    prod1 = (U[..., 1, :]*V[..., 1, :]).sum(1).unsqueeze(-1)
    prod2 = (U[..., 2, :]*V[..., 2, :]).sum(1).unsqueeze(-1)

    omega0 = torch.acos(prod0.clamp(-0.9999, 0.9999))
    omega1 = torch.acos(prod1.clamp(-0.9999, 0.9999))
    omega2 = torch.acos(prod2.clamp(-0.9999, 0.9999))

    res = torch.zeros_like(U_copy)
    res[..., 0, :] = (torch.sin((1.0-val)*omega0)*U_copy[..., 0, :] + torch.sin(val*omega0) * V_copy[..., 0, :]) / torch.sin(omega0)
    res[..., 1, :] = (torch.sin((1.0-val)*omega1)*U_copy[..., 1, :] + torch.sin(val*omega1) * V_copy[..., 1, :]) / torch.sin(omega1)
    res[..., 2, :] = (torch.sin((1.0-val)*omega2)*U_copy[..., 2, :] + torch.sin(val*omega2) * V_copy[..., 2, :]) / torch.sin(omega2)

    return res


def get_halfspace_mask(res):
    half1 = torch.ones((res, res//2), dtype=torch.bool)
    half2 = torch.zeros((res, res//2), dtype=torch.bool)
    return torch.cat((half1, half2), dim=-1).view(-1)


def get_facemask(img: torch.Tensor):

    imgnp = (img.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)

    landmarks = get_silhouette_lm(imgnp, method="dlib").astype(np.int32)

    mask = np.zeros(img.shape, dtype=np.uint8)
    mask = cv2.fillPoly(
        mask, np.array([landmarks], dtype=np.int32), (255, 255, 255)
    )
    cv2.imwrite("mask.png", mask)
    kernel = np.ones((5, 5), np.uint8)
    mask_eroded = cv2.erode(mask, kernel, iterations=3)
    maskt = torch.from_numpy(mask_eroded[:, :, 0].astype(bool)).view(-1)
    return maskt
    # cv2.imwrite("test.png", mask)
    # br = cv2.boundingRect(mask[:, :, 0])
    # center = (int(br[0] + br[2] / 2), int(br[1] + br[3] / 2))


def get_botton_facemask(img: torch.Tensor):

    imgnp = (img.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)

    landmarks = get_bottom_face_lm(imgnp).astype(np.int32)

    mask = np.zeros(img.shape, dtype=np.uint8)
    mask = cv2.fillPoly(
        mask, np.array([landmarks], dtype=np.int32), (255, 255, 255)
    )
    cv2.imwrite("mask.png", mask)
    kernel = np.ones((5, 5), np.uint8)
    mask_eroded = cv2.erode(mask, kernel, iterations=3)
    maskt = torch.from_numpy(mask_eroded[:, :, 0].astype(bool)).view(-1)
    return maskt
    # cv2.imwrite("test.png", mask)
    # br = cv2.boundingRect(mask[:, :, 0])
    # center = (int(br[0] + br[2] / 2), int(br[1] + br[3] / 2))


def get_mask(img, points, erosions=3):
    imgnp = (img.detach().cpu().numpy() * 255).astype(np.uint8)
    landmarks = ((points / 2. + 0.5) * img.shape[0]).astype(np.int32)
    landmarks = np.fliplr(landmarks)

    mask = np.zeros(imgnp.shape, dtype=np.uint8)
    mask = cv2.fillPoly(
        mask, np.array([landmarks], dtype=np.int32), (255, 255, 255)
    )
    # cv2.imwrite("mask.png", mask)
    if erosions:
        kernel = np.ones((5, 5), np.uint8)
        mask_eroded = cv2.erode(mask, kernel, iterations=erosions)
        cv2.imwrite("mask_eroded.png", mask_eroded)
        return torch.from_numpy(mask_eroded[:, :, 0].astype(bool)).view(-1)

    return torch.from_numpy(mask[:, :, 0].astype(bool)).view(-1)


def cv_blending(src, tgt, mask, blending_type=cv2.NORMAL_CLONE):
    rec1 = tgt

    rec0np = (src.detach().cpu().clamp(0, 1) * 255).numpy().astype(np.uint8)
    rec1np = (rec1.detach().cpu().clamp(0, 1) * 255).numpy().astype(np.uint8)
    masknp = (mask.detach().cpu().clamp(0, 1) * 255).numpy().astype(np.uint8)

    br = cv2.boundingRect(masknp)
    center = (int(br[0] + br[2] / 2), int(br[1] + br[3] / 2))

    rec = cv2.seamlessClone(
        rec1np, rec0np, masknp, p=center, flags=blending_type
    )
    return rec


class GradientMix(Enum):
    TARGET_TO_SOURCE = "target2source"
    SOURCE_TO_TARGET = "source2target"
    AVG_CLONE = "avgclone"
    MIX_CLONE = "mixclone"


class BlendingType(Enum):
    NEURAL = "neural"
    OPENCV = "opencv"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Neural (or classic) poisson blending of two images."
    )
    parser.add_argument(
        "config_path", help="Path to experiment configuration"
        " file."
    )
    parser.add_argument(
        "--seed", default=123, type=int,
        help="Seed for the RNG. By default its 123."
    )
    # parser.add_argument(
    #     "--n-tasks", default=1, type=int,
    #     help="Number of parallel trainings to run. By default is set to 1,"
    #     " meaning that we run serially."
    # )
    # parser.add_argument(
    #     "--skip-finished", action="store_true",
    #     help="Skips running an experiment if the output path contains the"
    #     " \"weights.pth\" file."
    # )
    # parser.add_argument(
    #     "--output-path", "-o", default="results",
    #     help="Optional output path to store experimental results. By default"
    #     " we use the experiment filename and create a matching directory"
    #     " under folder \"results\"."
    # )
    parser.add_argument(
        "--landmark-model", default="", help="The landmark model to use when"
        " building the mask that defines the blending region. May be empty"
        " (default), \"dlib\", \"mediapipe\", \"spiga\"."
    )
    parser.add_argument(
        "--blending", default="neural", type=str, help="Which blending to"
        " perform: \"opencv\" or \"neural\"?"
    )
    parser.add_argument(
        "--image-size", default="768x768", type=str, help="Size of output "
        " image in pixels. Note that we accept only the following format:"
        " HEIGHTxWIDTH, e.g. 768x768 (default)."
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    output_path = osp.join(
        osp.split(osp.join(config["warp_model"]))[0],
        "poisson-blending"
    )

    os.makedirs(output_path, exist_ok=True)

    device = torch.device(config["device"])
    trainingcfg = config["training"]
    batch_size = trainingcfg["batch_size"]

    blending = BlendingType(args.blending)
    mix_type = GradientMix.SOURCE_TO_TARGET
    use_as_bg = "image1"
    using_dlib_face_mask = True

    # target images and warping
    model1 = from_pth(config["initial_conditions"][0], w0=1, device=device)
    model2 = from_pth(config["initial_conditions"][1], w0=1, device=device)
    warp_model = from_pth(config["warp_model"], w0=1, device=device)
    height, width = [int(d) for d in args.image_size.split('x')]
    
    # times
    T = config["loss"]["intermediate_times"]
    grid = get_grid((height, width)).to(device)

    for i, t in enumerate(T):
        #grid2d = get_grid((height, width)).to(device)
        time_coord = torch.full((height * width, 1), t).to(device)
        coords = torch.cat([grid, time_coord], dim=-1)

        jac_blending = []
        gt_img1 = []
        gt_img2 = []

        num_steps = 2 * math.ceil(grid.shape[0] / batch_size)
        npoints1 = math.ceil(coords.shape[0] / num_steps)
        for step in range(num_steps):
            batched_coords = coords[step * npoints1:(step+1) * npoints1, ...].detach().clone().requires_grad_(True)

            # warped_image1(x,t) = I1 o T(x,-t). Paper notation.
            batched_coords1 = torch.cat((batched_coords[..., :2], -batched_coords[..., -1:]), dim=-1).to(device)
            warped_coords = warp_model(batched_coords1, preserve_grad=True)["model_out"]
            image1 = model1(warped_coords.to(device), preserve_grad=True)["model_out"]

            # grayscale1 = 0.2126 * output1[..., 0:1] + 0.7152 * output1[..., 1:2] + 0.0722 * output1[..., 2:3]
            # grad_img1 = gradient(grayscale1, gridt)[...,0:2].detach()

            jac1 = jacobian(image1.unsqueeze(0), batched_coords)[0].detach().to(device).squeeze(0)

            # warped_image2(x,t) = I2 o T(x,1-t). Paper notation.
            grid3d2 = batched_coords.detach().clone().to(device).requires_grad_(True)
            
            grid2 = torch.cat((grid3d2[..., :2], 1-grid3d2[..., -1:]), dim=-1).to(device)
            wgrid2 = warp_model(grid2, preserve_grad=True)['model_out']
            image2 = model2(wgrid2.to(device), preserve_grad=True)["model_out"]

            # grayscale2 = 0.2126 * output2[..., 0:1] + 0.7152 * output2[..., 1:2] + 0.0722 * output2[..., 2:3]
            # grad_img2 = gradient(grayscale2, gridt)[...,0:2].detach()

            jac2 = jacobian(image2.unsqueeze(0), grid3d2)[0].detach().to(device).squeeze(0)

            norm1 = torch.norm(jac1,  p='fro', dim=[1, 2], keepdim=True)
            norm2 = torch.norm(jac2,  p='fro', dim=[1, 2], keepdim=True)

            # how to mix the gradients?
            if mix_type == GradientMix.MIX_CLONE:
                jac = (torch.where(norm1 < norm2, jac2, jac1))  # mixed cloning
            elif mix_type == GradientMix.SOURCE_TO_TARGET:
                jac = jac1  # clone image1 into image2
            elif mix_type == GradientMix.TARGET_TO_SOURCE:
                jac = jac2  # clone image2 into image1
            elif mix_type == GradientMix.AVG_CLONE:
                jac = (1-t)*jac1 + t*jac2  # average approach
            else:
                raise ValueError("Unknown type of blending.")

            jac_blending.append(jac.detach())
            gt_img1.append(image1.detach())
            gt_img2.append(image2.detach())

            jac1 = jac1.detach()
            jac2 = jac2.detach()

        jac_blending = torch.cat(jac_blending, dim=0).detach().to(device)

        gt_img1 = torch.cat(gt_img1, dim=0).detach().to(device).unsqueeze(0)
        gt_img2 = torch.cat(gt_img2, dim=0).detach().to(device).unsqueeze(0)

        # saving the warped images
        for j, wimg in enumerate([gt_img1, gt_img2]):
            wimg = wimg.reshape(height, width, 3).cpu().clamp(0, 1).numpy() * 255
            cv2.imwrite(
                osp.join(output_path, f"warped_{j}.png"),
                cv2.cvtColor(
                    wimg.astype(np.uint8),
                    cv2.COLOR_RGB2BGR
                )
            )

        # Setting the background image according to how we mixed the gradients
        if mix_type == GradientMix.SOURCE_TO_TARGET:
            gt_img = gt_img2
        elif mix_type == GradientMix.TARGET_TO_SOURCE:
            gt_img = gt_img1
        else:
            if use_as_bg == "image1":
                gt_img = gt_img1
            elif use_as_bg == "image2":
                gt_img = gt_img2
            else:
                gt_img = 0.5 * (gt_img1 + gt_img2)  # you can define a background image here!

        if args.landmark_model == "dlib":
            mask = get_facemask(gt_img.reshape(height, width, 3)).to(device)
            # mask = get_halfspace_mask(res).to(device)
        else:
            # creting a mask by hand
            gt_blend = 0.5*(gt_img1+gt_img2)
            ui = FaceInteractor(
                gt_blend.reshape(height, width, 3).cpu().numpy(),
                gt_img2.reshape(height, width, 3).cpu().numpy()
            )
            plt.show()
            src_points, tgt_points = ui.landmarks
            mask = get_mask(gt_img.reshape(height, width, 3), src_points, erosions=0)

        if blending == BlendingType.OPENCV:
            for bt, btstr in zip([cv2.MIXED_CLONE, cv2.NORMAL_CLONE], ["mixed", "avg"]):
                rec = cv_blending(
                    gt_img1.reshape(height, width, 3),
                    gt_img2.reshape(height, width, 3),
                    t,
                    mask.reshape(height, width),
                    blending_type=bt
                )
                cv2.imwrite(
                    osp.join(output_path, f"ocv_no_warp_{btstr}.png"),
                    cv2.cvtColor(rec, cv2.COLOR_RGB2BGR)
                )
        elif blending == BlendingType.NEURAL:
            mask_copy = mask.detach().cpu().numpy().astype(np.uint8) * 255
            mask_copy = cv2.erode(mask_copy, np.ones((5, 5)), iterations=4)
            mask_copy = torch.from_numpy(mask_copy.astype(bool)).view(-1).to(device)

            dataset = PoissonEqn(gt_img, jac_blending, grid.detach())

            netcfg = config["network"]
            mmodel = SIREN(
                n_in_features=netcfg["in_channels"],
                n_out_features=netcfg["out_channels"],
                hidden_layer_config=netcfg["hidden_layers"],
                w0=netcfg["omega_0"],
                ww=netcfg.get("omega_w", netcfg["omega_0"])
            )
            mmodel.to(device)
            constraint_weights = dict((k, float(c)) for k, c in config["loss"]["constraint_weights"].items())
            loss_fn = GradientBlendingLoss(
                constraint_weights=constraint_weights
            )

            total_steps = trainingcfg["n_steps"]
            steps_til_summary = trainingcfg["checkpoint_steps"]

            optim = torch.optim.Adam(
                lr=config["optimizer"]["lr"],
                params=mmodel.parameters()
            )

            model_input, gt = dataset[0]
            gt = {key: value.to(device) for key, value in gt.items()}

            N = math.ceil(grid.shape[0] / batch_size)
            npoints = math.ceil(coords.shape[0] / N)
            for epoch in range(total_steps):
                idx = torch.randperm(grid.shape[0], device=device)
                for step in range(N):
                    start_time = time.time()

                    idxslice = idx[step*batch_size:(step+1)*batch_size]

                    in_mask_output = mmodel(grid[idxslice, ...][mask[idxslice]].unsqueeze(0))
                    outside_mask_output = mmodel(grid[idxslice, ...][~mask_copy[idxslice]].unsqueeze(0))

                    loss = loss_fn(
                        in_mask_output,
                        outside_mask_output,
                        gt["jac"][idxslice, ...][mask[idxslice]],
                        gt_img[:, idxslice, ...][~mask_copy[idxslice].unsqueeze(0)]
                    )

                    train_loss = torch.zeros((1, 1)).to(device)
                    for k, v in loss.items():
                        train_loss += v

                    optim.zero_grad()
                    train_loss.backward()
                    optim.step()

                if epoch and not epoch % steps_til_summary:
                    print("epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
                    mmodel = mmodel.eval()
                    with torch.no_grad():
                        grid = grid.detach()
                        output_dict = mmodel(grid)
                        rec_image = output_dict["model_out"]

                        img = rec_image.detach().clamp(0, 1).view(height, width, 3).cpu().numpy() * 255
                        cv2.imwrite(
                            osp.join(output_path, f"rec-{epoch}_t-{t}.png"),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        )
                    mmodel = mmodel.train()

            mmodel = mmodel.eval()
            with torch.no_grad():
                mmodel.update_omegas(w0=1, ww=None)
                torch.save(
                    mmodel.state_dict(),
                    osp.join(output_path, f"blending_weights_t-{t}.pth")
                )
                rec_image = mmodel(grid.detach())["model_out"]

                img = rec_image.detach().clamp(0, 1).view(height, width, 3).cpu().numpy() * 255
                cv2.imwrite(
                    osp.join(output_path, f"final_t-{t}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                )
