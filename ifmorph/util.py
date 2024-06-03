# coding: utf-8

import math
import os.path as osp
import cv2
import dlib
import numpy as np
import torch
import matplotlib.pyplot as plt
import mediapipe as mp
from ifmorph.point_editor import get_mediapipe_coord_dict
from ifmorph.point_editor import FaceInteractor

mp_face_mesh = mp.solutions.face_mesh


def get_dlib_coord_dict():
    return {
        "sillhouete": list(range(0, 17)),
        "left_eyebrow": list(range(22, 27)),
        "right_eyebrow": list(range(17, 22)),
        "nose": list(range(27, 31)),
        "lower_nose": list(range(31, 36)),
        "left_eye": list(range(42, 48)),
        "right_eye": list(range(36, 42)),
        "upper_lip_outer": list(range(48, 55)),
        "lower_lip_outer": list(range(55, 51)),
        "upper_lip_inner": [61, 62, 63],
        "lower_lip_inner": [65, 66, 67],
    }


def get_landmarks_dlib(img):
    """Uses DLib 68 point model to find face landmarks in `img`.

    Parameters
    ----------
    img: np.array
        The image containing the face. We didn't test with multiple faces.

    Returns
    -------
    landmarks: np.array
        An [N, 2] shaped array with coordinates of the landmarks in `img`.

    Raises
    ------
    ValueError: if no faces are detected
    """
    detector_dlib = dlib.get_frontal_face_detector()
    predictor_dlib = dlib.shape_predictor(
        osp.join("landmark_models", "shape_predictor_68_face_landmarks_GTX.dat")
    )
    landmarks = []
    try:
        facerect = detector_dlib(img, 1)[0]
    except IndexError:
        print("No faces detected.")
        raise ValueError("No faces detected.")
    shape = predictor_dlib(img, facerect)
    for i in range(shape.num_parts):
        p = shape.part(i)
        landmarks.append((p.y, p.x))

    landmarks = np.array(landmarks)
    return landmarks


def get_grid(dims, requires_grad=False, list_of_coords=True):
    """Returns a `dims`-shaped grid.

    Parameters
    ----------
    dims: number, tuple[numbers]
        Dimensions of the output grid

    requires_grad: boolean, optional
        Set to True if you want to retain the gradient of the output grid.
        Default value is False.

    list_of_coords: boolean, optional
        Set to False to return the dims shaped grid, or keep it True to
        return a prod(dims), len(dims) shaped tensor, i.e., a list of
        coordinates. Default value is True.

    Returns
    -------
    coords: torch.Tensor
        Note that the coordinates will always be in range [-1, 1].
    """
    if isinstance(dims, int):
        dims = [dims]
    tensors = tuple([torch.linspace(-1, 1, steps=d) for d in dims])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    if list_of_coords:
        mgrid = mgrid.reshape(-1, len(dims))
    mgrid.requires_grad_(requires_grad)
    return mgrid


def get_silhouette_lm(img, method="mediapipe"):
    """Returns the silhouette landmarks from `img` in pixel coordinates.

    Parameters
    ----------
    img: numpy.ndarray
        The input image with shape [H, W, C] with value range in [0, 255].

    method: str, optional
        The face detection/landmark method to use. May be either "mediapipe"
        (default), or "dlib".

    Returns
    -------
    landmarks: list
        A list of pairs of coordinates with locations of each landmark in image
        space.
    """
    landmarks = []
    if method == "mediapipe":
        sillhoueteidx = get_mediapipe_coord_dict()["silhouette"]
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            min_detection_confidence=0.75
        )
        mesh = face_mesh.process(img)
        landmarks = mesh.multi_face_landmarks[0].landmark
        landmarks_list = [[landmark.x, landmark.y] for landmark in landmarks]
        landmarks = np.array(landmarks_list)[sillhoueteidx]
        landmarks[:, 0] *= img.shape[0]
        landmarks[:, 1] *= img.shape[1]
    elif method == "dlib":
        coord_dict = get_dlib_coord_dict()
        maskpts = coord_dict["sillhouete"]
        maskpts.extend(coord_dict["left_eyebrow"][::-1])
        maskpts.extend(coord_dict["right_eyebrow"][::-1])

        detector_dlib = dlib.get_frontal_face_detector()
        predictor_dlib = dlib.shape_predictor(
            osp.join("landmark_models", "shape_predictor_68_face_landmarks_GTX.dat")
        )
        try:
            faces = detector_dlib(img, 1)
            facerect = faces[0]
        except IndexError:
            print("No faces detected.")
            raise
        shape = predictor_dlib(img, facerect)
        for i in range(shape.num_parts):
            p = shape.part(i)
            landmarks.append((p.x, p.y))
        landmarks = np.array(landmarks)[maskpts]

    return landmarks


def get_bottom_face_lm(img):
    landmarks = []
    coord_dict = get_dlib_coord_dict()
    maskpts = coord_dict["sillhouete"][1:-1]
    maskpts.extend(coord_dict["lower_nose"][::-1])

    detector_dlib = dlib.get_frontal_face_detector()
    predictor_dlib = dlib.shape_predictor(
        osp.join("landmark_models", "shape_predictor_68_face_landmarks_GTX.dat")
    )
    try:
        faces = detector_dlib(img, 1)
        facerect = faces[0]
    except IndexError:
        print("No faces detected.")
        raise
    shape = predictor_dlib(img, facerect)
    for i in range(shape.num_parts):
        p = shape.part(i)
        landmarks.append((p.x, p.y))
    landmarks = np.array(landmarks)[maskpts]

    return landmarks


def plot_scalars(t: torch.Tensor):
    """Plots a batch of the input tensors as images. The input tensors must
    contain scalar values, e.g. their shape must be 1xHxW

    Parameters
    ----------
    t: torch.Tensor
        The input tensor with shape Bx1xHxW.

    Returns
    -------
    fig: matplotlib.Figure
    ax: matplotlib.Axes
    """
    N = t.shape[0]
    rows = math.ceil(math.sqrt(N))
    cols = math.ceil(N / rows)
    fig, ax = plt.subplots(rows, cols, sharex=True, sharey=True)
    if isinstance(ax, np.ndarray):
        ax = ax.flatten()
        for i in range(N):
            img = t[i, ...].permute(1, 2, 0)
            ax[i].imshow(img, cmap="gray")
            ax[i].axis("off")
            ax[i].set_xticks([])
            ax[i].set_yticks([])
    else:
        img = t[0, ...].permute(1, 2, 0)
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    return fig, ax


def batched_predict(model: torch.nn.Module, coord_arr: torch.Tensor,
                    batch_size: int):
    """Runs model prediction in batches to avoid memory issues."""
    with torch.no_grad():
        n = coord_arr.shape[0]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + batch_size, n)
            yhat = model(coord_arr[ql:qr, :])["model_out"]
            preds.append(yhat)
            ql = qr
        preds = torch.cat(preds, dim=0)
    return preds


def warped_shapenet_inference(
        grid_with_t, warpnet, shapenet, framedims, rot_angle=0,
        translation=(0, 0), bggray=0
):
    """Performs the inference on a `warpnet` and feeds the results to a
    `shapenet`.

    Parameters
    ----------
    grid_with_t: torch.Tensor

    warpnet: torch.nn.Module
        A network that performs the warping of coordinates in `grid_with_t`.
        Must accept a [N, 3] tensor as input, and outputs a [N, 2] tensor.

    shapenet: torch.nn.Module
        A network that receives the [N, 2] output of `flownet` and runs the
        inference on those points. It must output a an Nx{1,3} tensor,
        depending on the number of channels of the output image.

    framedims: tuple with *2* ints
        The number of rows and columns of the output image. Note that the
        number of channels is inferred from `shapenet`.

    Returns
    -------
    img: torch.Tensor
        The resulting image with shape `(framedims[0], framedims[1],
        shapenet.out_features)`

    coords: torch.Tensor
        The warped coordinates used as input for `shapenet`.
    """
    coords = warpnet(grid_with_t)["model_out"].detach()
    if any(translation):
        coords -= -torch.tensor(translation).to(device=coords.device)
    if rot_angle:
        coords_copy = coords.clone()
        cos = torch.cos(torch.tensor(rot_angle))
        sin = torch.sin(torch.tensor(rot_angle))
        coords[:, 0] = cos * coords_copy[:, 0] + sin * coords_copy[:, 1]
        coords[:, 1] = -sin * coords_copy[:, 0] + cos * coords_copy[:, 1]

    img = shapenet(coords)["model_out"].detach().clamp(0, 1) * 255
    # restrict to the image domain [-1,1]^2
    img = torch.where(
        torch.abs(coords[..., 0].unsqueeze(-1)) < 1.0, img, torch.full_like(img, bggray)
    )
    img = torch.where(
        torch.abs(coords[..., 1].unsqueeze(-1)) < 1.0, img, torch.full_like(img, bggray)
    )
    img = img.reshape([framedims[0], framedims[1], shapenet.out_features])
    return img, coords


def warp_points(
        model: torch.nn.Module, points: torch.Tensor, t: float
) -> torch.Tensor:
    """Warps Nx2 `points` by parameter `t` \in [0, 1] using `model`.
    
    Parameters
    ----------
    model: torch.nn.Module
        A warping network with input Nx3.

    points: torch.Tensor
        Nx2 tensor of points to be warped.

    t: number
        Parameter in [0, 1] range to warp the points.

    Returns
    -------
    warped_points: torch.Tensor
    """
    t_points = torch.cat((points, torch.full_like(points[..., -1:], t)), dim=1)
    return model(t_points)["model_out"]


def plot_landmarks(im: np.array, lm: np.array, c=(0, 255, 0), r=3) -> np.array:
    """Overlays landmarks `lm` on the image `im` with colors `c` and radius
    `r`.
    
    Parameters
    ----------
    im: np.array
        The HxWx3 image.

    lm: np.array
        The Nx2 landmark coordinates. If they are in range [-1, 1] or [0, 1],
        we will normalize them using `im.shape`.

    c: tuple, optional
        The RGB landmark color.

    r: int, optional
        The landmark radius

    Returns
    -------
    imnp: np.ndarray
        The input image in numpy format with landmarks overlaid.
    """
    imc = im.copy()
    lmn = lm.copy()

    if lm.max() <= 1.0:  # assumed to be in [?, 1] range
        if lm.min() < 0:  # assumed to be in range [-1, 1]
            lmn = (lmn + 1.0) / 2.0
        lmn[:, 0] *= imc.shape[0]
        lmn[:, 1] *= imc.shape[1]
        lmn = lmn.astype(np.uint32)

    rad = r//2
    for l in lmn:
        imc[(l[0]-rad):(l[0]+rad), (l[1]-rad):(l[1]+rad), :] = c

    return imc


def blend_frames(f1: torch.Tensor, f2: torch.Tensor, t: float, blending_type: str) -> torch.Tensor:
    """Blends frames `f1` and `f2` following `blending_type`.

    Parameters
    ----------
    f1: torch.Tensor
        The [M,N] shaped initial frame.

    f2: torch.Tensor
        The [M,N] shaped final frame.

    t: float
        The time parameter for blending. For most blending types, `t=0` returns
        `f1` and `t=1` returns `f2`.

    blending_type: str
        The blending method. May be any one of: linear src, dst, min, max,
        seamless_{mix,clone}

    Returns
    -------
    blended_frame: torch.Tensor
    """
    if blending_type == "linear":
        rec = (1 - t) * f1 + t * f2
    elif "min" in blending_type:
        rec = torch.minimum(f1, f2)
    elif "max" in blending_type:
        rec = torch.maximum(f1, f2)
    # elif "dist" in blending_type:
    #     dist_0 = torch.sqrt(torch.sum((coords0[..., :2] - grid)**2, dim=-1)).unsqueeze(-1)
    #     dist_1 = torch.sqrt(torch.sum((coords1[..., :2] - grid)**2, dim=-1)).unsqueeze(-1)
    #     f1tmp = f1.reshape((dist_0.shape[0], shape_net0.out_features))
    #     f2tmp = f2.reshape((dist_1.shape[0], shape_net1.out_features))
    #     rec = (dist_1 * f1tmp + dist_0 * f2tmp) / (dist_0 + dist_1)
    #     rec = rec.reshape(f1.shape).detach().cpu().numpy().astype(np.uint8)
    elif "src" in blending_type:
        rec = f1
    elif "tgt" in blending_type:
        rec = f2
    elif "seamless" in blending_type:
        flags = cv2.NORMAL_CLONE
        if "mix" in blending_type:
            flags = cv2.MIXED_CLONE
            f2 = t * f2
        else:
            f2 = (1 - t) * f1 + t * f2

        f1np = f1.detach().cpu().numpy().astype(np.uint8)
        f2np = f2.detach().cpu().numpy().astype(np.uint8)

        landmarks = get_silhouette_lm(f1np, method="dlib").astype(np.int32)

        mask = np.zeros(f2.shape, dtype=np.uint8)
        mask = cv2.fillPoly(
            mask, np.array([landmarks], dtype=np.int32), (255, 255, 255)
        )
        br = cv2.boundingRect(mask[:, :, 0])
        center = (int(br[0] + br[2] / 2), int(br[1] + br[3] / 2))
        kernel = np.ones((5, 5), np.uint8)
        mask_eroded = cv2.erode(mask, kernel, iterations=1)

        rec = cv2.seamlessClone(
            f2np, f1np, mask_eroded, p=center, flags=flags
        )

    if isinstance(rec, torch.Tensor):
        rec = rec.detach().cpu().numpy().astype(np.uint8)

    return rec


def create_morphing_video(
        warp_net: torch.nn.Module,
        shape_net0: torch.nn.Module,
        shape_net1: torch.nn.Module,
        output_path: str,
        frame_dims: tuple,
        n_frames: int,
        fps: int,
        ic_times: list,
        device: torch.device,
        src,
        tgt,
        plot_landmarks=True,
        blending_type="linear",
        angles=(0, 0),
        translations=[[0, 0], [0, 0]]
):
    """Creates a video file given a model and output path.

    Parameters
    ----------
    warp_net: torch.nn.Module
        The warping model. It must be a coordinate model with three
        inputs (u, v, t), where u, v range in [-1, 1] and t is the time, which
        may have the same range as u, v, and is given by the `time_range`
        parameter.

    output_path: str, PathLike
        Output path to the video generated.

    frame_dims: tuple of ints
        The (width, height) of each video frame.

    n_frames: int
        The number of frames in the final video.

    fps: int
        Frames-per-second of the video.

    ic_times: list
        The times of the initial conditions.

    device: torch.device
        The device to run the inference for all networks. All intermediate
        tensors will be allocated with this device option.

    src: torch.Tensor
        Warping source points

    tgt: torch.Tensor
        Warping target points

    plot_landmarks: boolean, optional
        Switch to plot the warping points over the video (if True), or not
        (if False, default behaviour)

    blending_type: str, optional
        How to perform the blending of the initial states, may be "linear"
        (default), which performs a linear interpolation of the states.
        "minimum" and "maximum" get minimum(maximum) color values between the
        inferences of `shape_net0` and `shape_net1` at the warped coordinates.
        Finally, `dist` performs a distance based blending...

    angles: tuple, optional
        The rotation angles (in degrees) to apply to the source (angles[0]) and
         target (angles[1]) images. Useful for robustness test. By default its
        (0, 0), meaning no rotation applied to the images.

    Returns
    -------
    Nothing
    """
    warp_net = warp_net.eval()
    shape_net0 = shape_net0.eval()
    shape_net1 = shape_net1.eval()

    t1 = 0
    t2 = 1
    times = np.arange(t1, t2, (t2 - t1) / n_frames)
    grid = get_grid(frame_dims).to(device).requires_grad_(False)

    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps,
                          frame_dims[::-1], True)

    with torch.no_grad():
        if isinstance(src, torch.Tensor):
            src = src.clone().detach()
        else:
            src = torch.tensor(src, device=device, dtype=torch.float32)

        if isinstance(tgt, torch.Tensor):
            tgt = tgt.clone().detach()
        else:
            tgt = torch.tensor(tgt, device=device, dtype=torch.float32)

        tgrid = torch.hstack(
            (grid, torch.zeros((grid.shape[0], 1), device=device))
        ).requires_grad_(False)

        for t in times:
            tgrid[..., -1] = -t
            rec0, coords0 = warped_shapenet_inference(
                tgrid, warp_net, shape_net0, frame_dims, rot_angle=angles[0],
                translation=translations[0]
            )

            tgrid[..., -1] = 1 - t
            rec1, coords1 = warped_shapenet_inference(
                tgrid, warp_net, shape_net1, frame_dims, rot_angle=angles[1],
                translation=translations[1]
            )

            rec = blend_frames(rec0, rec1, t, blending_type=blending_type)
            rec = cv2.cvtColor(rec, cv2.COLOR_RGB2BGR)
            cv2.putText(
                img=rec, text='Time = %.2f' % (t.item()), org=(0, 30),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5,
                color=(255, 255, 255), thickness=1
            )

            if plot_landmarks:
                warped_src = list(
                    warp_points(warp_net, src, t).detach().cpu().numpy()
                )
                warped_tgt = list(
                    warp_points(warp_net, tgt, t - 1).detach().cpu().numpy()
                )
                for (point_tgt, point_src) in zip(warped_tgt, warped_src):
                    norm_src = (int(frame_dims[1]*(point_src[1]+1)/2),
                                int(frame_dims[0]*(point_src[0]+1)/2))
                    norm_tgt = (int(frame_dims[1]*(point_tgt[1]+1)/2),
                                int(frame_dims[0]*(point_tgt[0]+1)/2))
                    rec = cv2.circle(
                        rec, norm_src, radius=1, color=(255, 0, 0), thickness=-1
                    )
                    rec = cv2.circle(
                        rec, norm_tgt, radius=1, color=(0, 255, 0), thickness=-1
                    )
        
            out.write(rec)
    out.release()


def grid_image(coords):
    N = 8
    M = 8
    x = coords[..., 0].unsqueeze(-1)*N
    y = coords[..., 1].unsqueeze(-1)*M

    colors = torch.exp(-(x - torch.floor(x))**2/0.001)
    colors += torch.exp(-(y - torch.floor(y))**2/0.001)
    colors = 1 - torch.clamp(colors, 0, 1)

    return colors


def image_inference(model, grid_dims, device=torch.device("cpu")):
    """Runs inference on the model. We assume that the model represents an
    image.

    Parameters
    ----------
    model: torch.nn.Module
        The model to run inference on. The `forward` call must return a
        dictionary with a `model_out` key.

    grid_dims: collection of *two* ints
        (height, width) of the output image. We don't check for invalid
        values (<= 0).

    device: str or torch.Device, optional
        The device to run the inference on. We will create an evenly spaced
        grid, transfer the model and run the inference on this device. Note
        that we don't transfer the model back to its original device, nor
        check if the device is the same as the old one. Default value is
        `"cpu"`.

    Returns
    -------
    img: torch.Tensor
        The inference results with shape (grid_dims[0], grid_dims[1],
        n_channels), where n_channels is model.out_features.
    """
    model = model.eval().to(device)
    n_channels = model.out_features
    grid = get_grid(grid_dims).to(device)

    with torch.no_grad():
        img = (model(grid)["model_out"].detach().clamp(0, 1) * 255)
        img = img.cpu().reshape((grid_dims[0], grid_dims[1], n_channels)).numpy().astype(np.uint8)

    return img


def get_landmarks(face_mesh, img):
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    landmarks = results.multi_face_landmarks[0].landmark
    landmarks_list = [[landmark.y, landmark.x] for landmark in landmarks]

    larndmarks_np = np.array(landmarks_list)

    dict_face = {
        'silhouette': [10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                       397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                       172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109],

        'lipsUpperOuter':  [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
        'lipsLowerOuter': [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
        'lipsUpperInner': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
        'lipsLowerInner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],

        'rightEyeUpper0': [246, 161, 160, 159, 158, 157, 173],
        'rightEyeLower0': [33, 7, 163, 144, 145, 153, 154, 155, 133],
        # rightEyeUpper1: [247, 30, 29, 27, 28, 56, 190],
        # rightEyeLower1: [130, 25, 110, 24, 23, 22, 26, 112, 243],
        # rightEyeUpper2: [113, 225, 224, 223, 222, 221, 189],
        # rightEyeLower2: [226, 31, 228, 229, 230, 231, 232, 233, 244],
        # rightEyeLower3: [143, 111, 117, 118, 119, 120, 121, 128, 245],

        'rightEyebrowUpper': [156, 70, 63, 105, 66, 107, 55, 193],
        # "rightEyebrowLower": [35, 124, 46, 53, 52, 65],

        # 'rightEyeIris': [473, 474, 475, 476, 477],

        'leftEyeUpper0': [466, 388, 387, 386, 385, 384, 398],
        'leftEyeLower0': [263, 249, 390, 373, 374, 380, 381, 382, 362],
        # "leftEyeUpper1": [467, 260, 259, 257, 258, 286, 414],
        # "leftEyeLower1": [359, 255, 339, 254, 253, 252, 256, 341, 463],
        # "leftEyeUpper2": [342, 445, 444, 443, 442, 441, 413],
        # "leftEyeLower2": [446, 261, 448, 449, 450, 451, 452, 453, 464],
        # "leftEyeLower3": [372, 340, 346, 347, 348, 349, 350, 357, 465],

        'leftEyebrowUpper': [383, 300, 293, 334, 296, 336, 285, 417],
        # "leftEyebrowLower": [265, 353, 276, 283, 282, 295],

        # 'leftEyeIris': [468, 469, 470, 471, 472],

        'midwayBetweenEyes': [168],

        'noseTip': [1],
        'noseBottom': [2],
        'noseRightCorner': [98],
        'noseLeftCorner': [327],

        'rightCheek': [205],
        'leftCheek': [425]
    }

    list_final = []
    for (k, v) in dict_face.items():
        list_final = list_final + v

    return larndmarks_np[list_final]


def return_points_morph_mediapipe(shape_net_src, shape_net_tgt, frame_dims,
                                  device="cuda:0"):
    shape_net_src = shape_net_src.eval()
    shape_net_tgt = shape_net_tgt.eval()
    n_channels = shape_net_src.out_features

    grid = get_grid(frame_dims).to(device)

    with torch.no_grad():
        src_img = (shape_net_src(grid)["model_out"].detach().clamp(0, 1) * 255).cpu().reshape((frame_dims[0], frame_dims[1], n_channels)).squeeze().numpy().astype(np.uint8)
        tgt_img = (shape_net_tgt(grid)["model_out"].detach().clamp(0, 1) * 255).cpu().reshape((frame_dims[0], frame_dims[1], n_channels)).squeeze().numpy().astype(np.uint8)

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, min_detection_confidence=0.75
    )

    landmarks_src = get_landmarks(face_mesh, src_img)
    landmarks_tgt = get_landmarks(face_mesh, tgt_img)

    # Need to return to original coordinates
    landmarks_src = landmarks_src * 2 - 1
    landmarks_tgt = landmarks_tgt * 2 - 1

    return (landmarks_src, landmarks_tgt, src_img, tgt_img)


def return_points_morph_dlib(shape_net_src, shape_net_tgt, frame_dims,
                             device="cuda:0"):
    shape_net_src = shape_net_src.eval()
    shape_net_tgt = shape_net_tgt.eval()
    n_channels = shape_net_src.out_features

    grid = get_grid(frame_dims).to(device)

    with torch.no_grad():
        src_img = (shape_net_src(grid)["model_out"].detach().clamp(0, 1) * 255).cpu().reshape((frame_dims[0], frame_dims[1], n_channels)).squeeze().numpy().astype(np.uint8)
        tgt_img = (shape_net_tgt(grid)["model_out"].detach().clamp(0, 1) * 255).cpu().reshape((frame_dims[0], frame_dims[1], n_channels)).squeeze().numpy().astype(np.uint8)

    landmarks = [None, None]
    for i, img in enumerate([src_img, tgt_img]):
        landmarks[i] = get_landmarks_dlib(img).astype(float)
        landmarks[i][:, 0] /= frame_dims[0]
        landmarks[i][:, 1] /= frame_dims[1]
        landmarks[i] = landmarks[i] * 2 - 1

    return (landmarks[0], landmarks[1], src_img, tgt_img)


def return_points_morph(shape_net_src, shape_net_tgt, frame_dims,
                        src_pts=None, tgt_pts=None, device="cuda:0",
                        run_mediapipe=True):
    shape_net_src = shape_net_src.eval()
    shape_net_tgt = shape_net_tgt.eval()
    dims = (frame_dims[0], frame_dims[1], shape_net_src.out_features)

    grid = get_grid(frame_dims, requires_grad=False).to(device)
    with torch.no_grad():
        src_img = (shape_net_src(grid)["model_out"].detach().clamp(0, 1) * 255).cpu().reshape(dims).numpy().astype(np.uint8)
        tgt_img = (shape_net_tgt(grid)["model_out"].detach().clamp(0, 1) * 255).cpu().reshape(dims).numpy().astype(np.uint8)

        p = FaceInteractor(
            src_img, tgt_img, src_pts=src_pts, tgt_pts=tgt_pts,
            run_mediapipe=run_mediapipe
        )
        plt.show()

        src, tgt = p.return_points()

    return (src, tgt, src_img, tgt_img)


def test_mediapipe_landmark_detection(img):
    im_landmarks = imr.copy()
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, min_detection_confidence=0.75
    )
    landmarks = get_landmarks(face_mesh, img)
    landmarks[:, 0] *= img.shape[1]
    landmarks[:, 1] *= img.shape[0]
    landmarks = landmarks.astype(int)

    for p in landmarks:
        im_landmarks[p[0]-1:p[0]+1, p[1]-1:p[1]+1] = (0, 255, 0)
    return im_landmarks


def test_dlib_landmark_detection(img):
    im_landmarks = img.copy()
    landmarks = get_landmarks_dlib(img)
    for p in landmarks:
        im_landmarks[p[1]-1:p[1]+1, p[0]-1:p[0]+1] = (0, 255, 0)
    return im_landmarks


if __name__ == "__main__":
    imnames = [f"{i}_03.jpg".zfill(10) for i in range(1, 10)]
    for i, imname in enumerate(imnames):
        imgpath = osp.join("data", "frll", imname)
        print(imgpath)
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imr = cv2.resize(img, (600, 600), interpolation=cv2.INTER_AREA)

        imdlib = test_dlib_landmark_detection(imr)
        cv2.imwrite(f"{i}_landmarks_dlib.png", cv2.cvtColor(imdlib, cv2.COLOR_RGB2BGR))

        immp = test_mediapipe_landmark_detection(imr)
        cv2.imwrite(f"{i}_landmarks_mp.png", cv2.cvtColor(immp, cv2.COLOR_RGB2BGR))
