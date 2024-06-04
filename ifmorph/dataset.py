#!/usr/bin/env python
# coding: utf-8

import math
import os.path as osp
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from ifmorph.model import from_pth
from ifmorph.util import get_grid

WITH_MRNET = True
try:
    from ext.mrimg.src.networks.mrnet import MRFactory
except (ModuleNotFoundError, ImportError):
    WITH_MRNET = False


def check_network_type(state_dict_path):
    """Returns the type of network stored in `state_dict_path`.

    Parameters
    ----------
    state_dict_path: str, PathLike
        Path to the PTH file with the network.

    Returns
    -------
    network_type: str
        Returns "siren" or "mrnet" according to the network type stored in the
        input file. Note that we assume it will be either a SIREN of one of the
        MRNet type networks, so we don't really check for other types here.
    """
    sd = torch.load(state_dict_path, map_location="cpu")
    return "mrnet" if "stages" in sd else "siren"


class NoMrnetError(Exception):
    """Exception raised when MRNet module is absent.

    Parameters
    ----------
    message: str
        Explanation of the error.
    """
    def __init__(self, message="MRNet is not present."):
        self.message = message
        super().__init__(self.message)


class ImageDataset(Dataset):
    """A Simple image dataset.

    Parameters
    ----------
    path: str
        Path to the image, should be readable by PIL .

    sidelen: int, optional
        Grid width and height. If set to -1 (which is the default case),
        will infer it from the image size.

    channels_to_use: int, optional
        The number of channels of the input image to use during training. Must
        be 1 for grayscale or 3 for RGB images. Default is None, meaning that
        the all channels of the input image will be used, whether its 1 or 3.

    batch_size: int, optional, NOT USED
        Number of samples to fetch at each call to __getitem__. Default is
        None, meaning that the whole image will be returned. Useful for
        memory-constrained scenarios.

    Raises
    ------
    ValueError:
        If `len(sidelen)` > 2 or `len(sidelen)` == 0. Also if image does
        not have the correct mode. Accepted modes are: "RGB", "L" and "P".

    TypeError:
        If `type(sidelen)` not in [tuple, list, int]
    """
    def __init__(self, path: str, sidelen=-1, channels_to_use=None,
                 batch_size=None):
        super(ImageDataset, self).__init__()
        if path[0] == "~":
            path = osp.expanduser(path)

        img = Image.open(path)
        grid_dims = None
        if sidelen != -1:
            if isinstance(sidelen, int):
                grid_dims = [sidelen] * 2
            elif isinstance(sidelen, (tuple, list)):
                if len(sidelen) > 2:
                    raise ValueError("sidelen has too many coordinates for"
                                     " image.")
                elif not sidelen:
                    raise ValueError("No grid size provided.")
                grid_dims = sidelen
            else:
                raise TypeError("sidelen is neither number or collection of"
                                " numbers.")
        else:
            grid_dims = (img.height, img.width)

        self.size = grid_dims

        if img.mode not in ["RGB", "L", "P"]:
            raise ValueError("Image must be RGB (3 channels) or grayscale"
                             f" (1 channel). # channels found: {len(img.mode)}"
                             f", format: {img.mode}")

        if channels_to_use is None:
            self.n_channels = 3 if img.mode == "RGB" else 1
        else:
            if channels_to_use not in [1, 3]:
                raise ValueError("Invalid number of channels to use. Should"
                                 f" be 1 or 3, found {channels_to_use}.")
            self.n_channels = channels_to_use

        t = [
            transforms.Resize(grid_dims),
            transforms.ToTensor()
        ] if sidelen != -1 else [transforms.ToTensor()]
        if self.n_channels == 1:
            t.append(transforms.Grayscale())

        t = transforms.Compose(t)
        self.coords = get_grid(grid_dims)
        self.pixels = t(img).permute(1, 2, 0).view(-1, self.n_channels)
        if not batch_size:
            self.batch_size = self.pixels.shape[0]
        else:
            self.batch_size = batch_size

    def __len__(self):
        return self.pixels.shape[0] // self.batch_size

    def __getitem__(self, idx=None):
        if idx is None or not len(idx):
            idx = torch.randint(self.coords.shape[0], (self.batch_size,))
        return self.coords[idx, ...], self.pixels[idx, ...], idx


class WarpingDataset(Dataset):
    """Warping dataset.

    Parameters
    ----------
    initial_states: list[tuples[number, torch.nn.Module]]
        A list of initial states (known images), and the time of each image.
        Note that the time should be in range [-1, 1]. And we will only sample
        in the time-range given here, i.e., if this parameter is
        [(-0.5, ...), (0.8, ...)], we will only sample values in range
        [-0.5, 0.8].

    num_samples: int
        Number of samples to draw at every call to __getitem__. Note that half
        of the samples will be drawn at the initial states (evenly distributed
        between them), and the other half will be drawn at intermediate times.

    device: str, torch.Device, optional
        The device to store the read models. By default is cpu.

    grid_sampling: boolean, optional
        Set to `True` (default) to sample points in a grid, distributed
        uniformely along time, or `False` to randomly sample points in the
        [-1, 1] domain.

    Examples
    --------
    > # Creating a dataset with 3 initial states at times -0.8, 0.4 and 0.95,
    > # all on CPU. We will fetch 1000 points per call to __getitem__.
    > initial_states = [("m1.pth", -0.8), ("m2.pth", 0.4), ("m3.pth", 0.95)]
    > data = WarpingDataset(initial_states, 1000, torch.device("cpu"))
    > X = data[0]
    > print(X.shape)  # Should print something like: [1000, 3]
    """
    def __init__(self, initial_states: list, num_samples: int,
                 device: str = "cpu", grid_sampling: bool = True):
        super(WarpingDataset, self).__init__()
        self.num_samples = num_samples
        self.device = device
        self.grid_sampling = grid_sampling
        self.initial_states = [None] * len(initial_states)
        self.known_times = [None] * len(initial_states)
        self.time_range = [-1.0, 1.0]
        for i, (state_path, t) in enumerate(initial_states):
            if check_network_type(state_path) == "siren":
                self.initial_states[i] = from_pth(
                    state_path, w0=1, device=device
                )
            else:
                if WITH_MRNET:
                    net = MRFactory.load_state_dict(state_path)
                    self.initial_states[i] = net.to(device)
                else:
                    raise NoMrnetError()
            self.known_times[i] = t

        # Spatial coordinates
        if self.grid_sampling:
            N = self.num_samples // 2
            m = int(math.sqrt(N))
            self.coords = get_grid([m, m]).to(self.device)
            self.int_times = 2 * (torch.arange(0, N, 1, device=self.device, dtype=torch.float32) - (N / 2)) / N

    @property
    def initial_conditions(self):
        return list(zip(self.initial_states, self.known_times))

    def __len__(self):
        return 1

    def __getitem__(self, _):
        """
        Returns
        -------
        X: torch.Tensor
            A [`num_samples`, 3] shaped tensor with the pixel coordinates at
            the first two columns, and time coordinates at the last column.

        y: torch.Tensor
            A [`num_samples`] shaped tensor with the image values at the
            initial conditions, and a -1 marker for intermediate times.
        """
        # # Spatial coordinates
        N = self.num_samples // 2

        if self.grid_sampling:
            int_times = self.int_times[torch.randperm(N, device=self.device)]
        else:
            self.coords = torch.rand((N, 2), device=self.device) * 2 - 1
            # Temporal coordinates \in (0, 1), renormalized to the actual time
            # ranges of the initial conditions.
            t1, t2 = self.time_range
            int_times = torch.rand(N, device=self.device) * (t2 - t1) + t1

        X = torch.cat([
            torch.cat((self.coords, torch.full_like(int_times, 0).unsqueeze(1).to(self.device)), dim=1)
        ], dim=0)
        X = torch.cat(
            (X, torch.hstack((self.coords, int_times.unsqueeze(1)))),
            dim=0
        )
        return X


class NoTimeWarpingDataset(WarpingDataset):
    """Warping dataset that doesn't return time coordinates.

    Parameters
    ----------
    initial_states: list[tuples[number, torch.nn.Module]]
        A list of initial states (known images), and the time of each image.
        Note that the time should be in range [-1, 1]. And we will only sample
        in the time-range given here, i.e., if this parameter is
        [(-0.5, ...), (0.8, ...)], we will only sample values in range
        [-0.5, 0.8].

    num_samples: int
        Number of samples to draw at every call to __getitem__. Note that half
        of the samples will be drawn at the initial states (evenly distributed
        between them), and the other half will be drawn at intermediate times.

    device: str, torch.Device, optional
        The device to store the read models. By default is cpu.

    time_range: list, optional
        Will sample points in this time_range independently from the timerange
        in `initial_states`.

    grid_sampling: boolean, optional
        Set to `True` (default) to sample points in a grid, distributed
        uniformely along time, or `False` to randomly sample points in the
        [-1, 1] domain.

    Examples
    --------
    > # Creating a dataset with 3 initial states at times -0.8, 0.4 and 0.95,
    > # all on CPU. We will fetch 1000 points per call to __getitem__.
    > initial_states = [("m1.pth", -0.8), ("m2.pth", 0.4), ("m3.pth", 0.95)]
    > data = NoTimeWarpingDataset(initial_states, 1000, torch.device("cpu"))
    > X, y = data[0]
    > print(X.shape, y.shape)  # Should print something like: [1000, 3], [1000]
    """
    def __init__(self, initial_states: list, num_samples: int,
                 device: str = "cpu", grid_sampling: bool = True):
        super(NoTimeWarpingDataset, self).__init__(
            initial_states, num_samples, device, grid_sampling
        )

    def __getitem__(self, _):
        """
        Returns
        -------
        X: torch.Tensor
            A [`num_samples`, 3] shaped tensor with the pixel coordinates at
            the first two columns, and time coordinates at the last column.
        """
        # Spatial coordinates
        N = self.num_samples // 2
        coords = torch.rand((N, 2), device=self.device) * 2 - 1
        return coords


if __name__ == "__main__":
    data = WarpingDataset([
        ("pretrained/yaleB01_P00A+000E+00.pth", -0.5),
        ("pretrained/yaleB11_P00A+000E+00.pth", 0.0),
        ("pretrained/yaleB27_P00A+000E+00.pth", 0.5)
    ], 48)
    X = data[0]
    print(X.shape)
