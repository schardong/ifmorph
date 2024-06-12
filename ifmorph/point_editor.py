# coding: utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton


def transform_display_coords(pts, frame_dims, is_tgt=False):
    x = (pts[:, 0] / 2. + 0.5) * frame_dims[0]
    y = (pts[:, 1] / 2. + 0.5) * frame_dims[1]

    if is_tgt:
        y = y + frame_dims[1]

    return y, x


def convert_coord_disp2src(pt, frame_dims, is_tgt=False):
    if is_tgt:
        pt[0] -= frame_dims[1]

    ptnew = [pt[0] / frame_dims[1], pt[1] / frame_dims[0]]
    ptnew = [2 * c - 1 for c in ptnew]
    return ptnew


def check_point_in_set(cursor_point, set_point, eps=15):
    d = np.hypot(
        set_point[0] - cursor_point[0],
        set_point[1] - cursor_point[1]
    )
    indseq, = np.nonzero(d == d.min())
    ind = indseq[0]

    if d[ind] >= eps:
        # print(f'rejeito com distancia: {d[ind]}')
        ind = None
    # else:
        # print(f'aceito com distancia: {d[ind]}')
        # print(cursor_point)
        # print(set_point[0][ind], set_point[1][ind])
        # print("conjunto certo")

    return ind


class FaceInteractor:
    """
    User-interface to mark/delete/edit landmark points on two faces.

    The final list of landmarks can be retrieved by calling `return_points`.

    Parameters
    ----------
    src_img: np.ndarray
        Source (leftmost) image

    tgt_img: np.ndarray
        Target (rightmost) image.

    eps: int, optional
        Tolerance value, for point operations (move/delete) in pixels. Default
         value is 15 pixels.

    src_pts: list, optional
        The source image landmarks. Optional, default is `None`.

    tgt_pts: list, optional
        The target image landmarks. Optional, default is `None`.
    """
    def __init__(self, src_img, tgt_img, eps: int = 15, src_pts=None,
                 tgt_pts=None):
        _, self.ax = plt.subplots()
        self.ax.set_title('Click and drag a point to move it')
        self.landmarks_src = None
        self.landmarks_tgt = None

        if src_pts is not None:
            if self.landmarks_src is not None:
                self.landmarks_src = np.vstack((self.landmarks_src, src_pts))
            else:
                self.landmarks_src = src_pts
        if tgt_pts is not None:
            if self.landmarks_tgt is not None:
                self.landmarks_tgt = np.vstack((self.landmarks_tgt, tgt_pts))
            else:
                self.landmarks_tgt = tgt_pts

        self.img_cat = np.concatenate([src_img, tgt_img], axis=1)
        self.frame_dims = (self.img_cat.shape[0], self.img_cat.shape[1]//2)
        self.sc_tgt = None
        self.sc_src = None

        canvas = self.ax.figure.canvas
        self._ind_tgt, self._ind_src = None, None
        self.eps = eps

        self.scatter_and_plot(
            self.landmarks_src, self.landmarks_tgt, self.frame_dims
        )
        canvas.mpl_connect('button_press_event', self.on_button_press)
        canvas.mpl_connect('button_release_event', self.on_button_release)
        canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas = canvas

    def scatter_and_plot(self, src_pts, tgt_pts, frame_dims):
        self.ax.clear()
        self.ax.imshow(self.img_cat)

        if src_pts is not None and len(src_pts):
            src_pts_norm = src_pts / 2.0 + 0.5
            src_pts_norm[:, 0] *= frame_dims[0]
            src_pts_norm[:, 1] *= frame_dims[1]
            self.sc_src = self.ax.scatter(
                src_pts_norm[:, 1], src_pts_norm[:, 0], s=5
            )

        if tgt_pts is not None and len(tgt_pts):
            tgt_pts_norm = tgt_pts / 2.0 + 0.5
            tgt_pts_norm[:, 0] *= frame_dims[0]
            tgt_pts_norm[:, 1] = tgt_pts_norm[:, 1] * frame_dims[1] + frame_dims[1]   # We need it to be on the target image
            self.sc_tgt = self.ax.scatter(
                tgt_pts_norm[:, 1], tgt_pts_norm[:, 0], c="green", s=5
            )

        plt.draw()

    def get_ind_under_point(self, event):
        """
        Return the index of the point closest to the event position or *None*
        if no point is within ``self.epsilon`` to the event position.
        """
        # display coords
        src_points_display = transform_display_coords(
            self.landmarks_src, self.frame_dims, is_tgt=False
        )
        tgt_points_display = transform_display_coords(
            self.landmarks_tgt, self.frame_dims, is_tgt=True
        )

        ind_src = check_point_in_set(
            (event.xdata, event.ydata),
            (src_points_display[0], src_points_display[1]),
            eps=self.eps
        )
        ind_tgt = check_point_in_set(
            (event.xdata, event.ydata),
            (tgt_points_display[0], tgt_points_display[1]),
            eps=self.eps
        )

        return ind_src, ind_tgt

    def on_button_press(self, event):
        """Callback for mouse button presses."""
        if event.inaxes is None:
            return

        x, y = event.xdata, event.ydata

        if event.button is MouseButton.LEFT:
            self._ind_src, self._ind_tgt = self.get_ind_under_point(event)

        if event.button is MouseButton.RIGHT:
            # print(event.button)
            ind_src_rem, ind_tgt_rem = self.get_ind_under_point(event)
            ind_rem = ind_src_rem if ind_tgt_rem is None else ind_tgt_rem
            if ind_rem is not None:
                self.landmarks_tgt = np.delete(self.landmarks_tgt, ind_rem, 0)
                self.landmarks_src = np.delete(self.landmarks_src, ind_rem, 0)
            self.scatter_and_plot(
                self.landmarks_src, self.landmarks_tgt, self.frame_dims
            )

        if event.button is MouseButton.MIDDLE:
            # print(event.button)
            if x <= self.frame_dims[1]:
                x, y = convert_coord_disp2src([x, y], self.frame_dims)
                newrow = [y, x]
                if self.landmarks_src is not None and len(self.landmarks_src):
                    self.landmarks_src = np.vstack([
                        self.landmarks_src, newrow
                    ])
                else:
                    self.landmarks_src = np.array([newrow])
            else:
                x, y = convert_coord_disp2src(
                    [x, y], self.frame_dims, is_tgt=True
                )
                newrow = [y, x]
                if self.landmarks_tgt is not None and len(self.landmarks_tgt):
                    self.landmarks_tgt = np.vstack([
                        self.landmarks_tgt, newrow
                    ])
                else:
                    self.landmarks_tgt = np.array([newrow])

            self.scatter_and_plot(
                self.landmarks_src, self.landmarks_tgt, self.frame_dims
            )

    def on_button_release(self, event):
        """Callback for mouse button releases."""
        if event.button != 1:
            return
        self._ind_src, self._ind_tgt = None, None

    def on_key_press(self, event):
        """Callback for key presses."""
        if not event.inaxes:
            return
        _, _ = event.xdata, event.ydata

    def on_mouse_move(self, event):
        """Callback for mouse movements."""
        if event.inaxes is None:
            return

        x, y = event.xdata, event.ydata

        if self._ind_src is not None:
            self.landmarks_src[self._ind_src, 1], self.landmarks_src[self._ind_src, 0] = convert_coord_disp2src([x, y], self.frame_dims)

        if self._ind_tgt is not None:
            self.landmarks_tgt[self._ind_tgt, 1], self.landmarks_tgt[self._ind_tgt, 0] = convert_coord_disp2src([x, y], self.frame_dims, True)

        self.scatter_and_plot(
            self.landmarks_src, self.landmarks_tgt, self.frame_dims
        )

    @property
    def landmarks(self):
        return (self.landmarks_src, self.landmarks_tgt)


if __name__ == '__main__':
    import os.path as osp

    fileimg1 = osp.join("data", "frll_neutral_front", "001_03.jpg")
    fileimg2 = osp.join("data", "frll_neutral_front", "002_03.jpg")

    img1 = cv2.cvtColor(cv2.imread(fileimg1), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(fileimg2), cv2.COLOR_BGR2RGB)

    img1 = cv2.resize(img1, (600, 600))
    img2 = cv2.resize(img2, (600, 600))

    p = FaceInteractor(img1, img2)
    plt.show()

    src, tgt = p.landmarks

    print(src)
    print(tgt)
