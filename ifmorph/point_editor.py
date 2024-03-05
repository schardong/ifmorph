# coding: utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh


def get_mediapipe_coord_dict():
    dict_face = {
        'silhouette': [
            10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
        ],

        'lipsUpperOuter':  [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
        'lipsLowerOuter': [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
        'lipsUpperInner': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
        'lipsLowerInner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],

        'rightEyeUpper0': [246, 161, 160],#, 159, 158, 157, 173],
        'rightEyeLower0': [33, 7, 163],#, 144, 145, 153, 154, 155, 133],
        #rightEyeUpper1: [247, 30, 29, 27, 28, 56, 190],
        #rightEyeLower1: [130, 25, 110, 24, 23, 22, 26, 112, 243],
        #rightEyeUpper2: [113, 225, 224, 223, 222, 221, 189],
        #rightEyeLower2: [226, 31, 228, 229, 230, 231, 232, 233, 244],
        #rightEyeLower3: [143, 111, 117, 118, 119, 120, 121, 128, 245],

        'rightEyebrowUpper': [156, 70, 63, 105, 66, 107, 55, 193],
        #rightEyebrowLower: [35, 124, 46, 53, 52, 65],

        #'rightEyeIris': [473, 474, 475, 476, 477],

        'leftEyeUpper0': [466, 388, 387],#, 386, 385, 384, 398],
        'leftEyeLower0': [263, 249, 390],# 373, 374, 380, 381, 382, 362],
        #leftEyeUpper1: [467, 260, 259, 257, 258, 286, 414],
        #leftEyeLower1: [359, 255, 339, 254, 253, 252, 256, 341, 463],
        #leftEyeUpper2: [342, 445, 444, 443, 442, 441, 413],
        #leftEyeLower2: [446, 261, 448, 449, 450, 451, 452, 453, 464],
        #leftEyeLower3: [372, 340, 346, 347, 348, 349, 350, 357, 465],

        'leftEyebrowUpper': [383, 300, 293, 334, 296, 336, 285, 417],
        #leftEyebrowLower: [265, 353, 276, 283, 282, 295],

        #'leftEyeIris': [468, 469, 470, 471, 472],

        'midwayBetweenEyes': [168],

        'noseTip': [1],
        'noseBottom': [2],
        'noseRightCorner': [98],
        'noseLeftCorner': [327],

        'rightCheek': [205],
        'leftCheek': [425]
    }
    return dict_face


class FaceInteractor:
    """
    A polygon editor.

    Key-bindings

      't' toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them

      'd' delete the vertex under point

      'i' insert a vertex at point.  You must be within epsilon of the
          line connecting two existing vertices

    """
    def get_landmarks(self, face_mesh, img):
        results = face_mesh.process(img)
        landmarks = results.multi_face_landmarks[0].landmark
        landmarks_list = [[landmark.y, landmark.x] for landmark in landmarks]

        larndmarks_np = np.array(landmarks_list)

        dict_face = get_mediapipe_coord_dict()

        list_final = []
        for v in dict_face.values():
            list_final = list_final + v

        return 2 * larndmarks_np[list_final] - 1

    def scatter_and_plot(self, src_pts, tgt_pts, frame_dims):
        ax = self.ax
        self.ax.clear()
        self.ax.imshow(self.img_cat)

        if src_pts is not None and len(src_pts):
            src_pts_norm = src_pts / 2.0 + 0.5
            src_pts_norm[:, 0] *= frame_dims[0]
            src_pts_norm[:, 1] *= frame_dims[1]
            self.sc_src = ax.scatter(src_pts_norm[:, 1], src_pts_norm[:, 0], s=5)

        if tgt_pts is not None and len(tgt_pts):
            tgt_pts_norm = tgt_pts / 2.0 + 0.5
            tgt_pts_norm[:, 0] *= frame_dims[0]
            tgt_pts_norm[:, 1] = tgt_pts_norm[:, 1] * frame_dims[1] + frame_dims[1]   # We need it to be on the target image
            self.sc_tgt = ax.scatter(tgt_pts_norm[:, 1], tgt_pts_norm[:, 0], c="green", s=5)

        plt.draw()

    def __init__(self, src_img, tgt_img, epsolon=15, src_pts=None,
                 tgt_pts=None, run_mediapipe=True):
        _, ax = plt.subplots()
        self.ax = ax
        self.ax.set_title('Click and drag a point to move it')

        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                          min_detection_confidence=0.5)

        self.landmarks_src = None
        self.landmarks_tgt = None
        if run_mediapipe:
            self.landmarks_src = self.get_landmarks(face_mesh, src_img)
            self.landmarks_tgt = self.get_landmarks(face_mesh, tgt_img)

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
        # print(self.frame_dims)
        self.sc_tgt = None
        self.sc_src = None

        self.ax = ax

        canvas = self.ax.figure.canvas
        self._ind_tgt, self._ind_src = None, None
        self.epsolon = epsolon

        self.scatter_and_plot(self.landmarks_src, self.landmarks_tgt, self.frame_dims)

        #canvas.mpl_connect('draw_event', self.on_draw)
        canvas.mpl_connect('button_press_event', self.on_button_press)
        canvas.mpl_connect('button_release_event', self.on_button_release)
        canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas = canvas

    #def on_draw(self, event):
        #self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        #self.ax.draw_artist(self.poly)
        #self.ax.draw_artist(self.line)
        # do not need to blit here, this will fire before the screen is
        # updated

    def poly_changed(self, poly):
        """This method is called whenever the pathpatch object is called."""
        pass
        # only copy the artist props to the line (except visibility)

        #vis = self.line.get_visible()
        #Artist.update_from(self.line, poly)
        #self.line.set_visible(vis)  # don't use the poly visibility state

    def transform_display_coords(self, pts, is_tgt=False):
        x = (pts[:, 0]/2.+0.5)*self.frame_dims[0]
        y = (pts[:, 1]/2.+0.5)*self.frame_dims[1]

        if is_tgt:
            y = y + self.frame_dims[1]
        return y, x

    def check_point_in_set(self, cursor_point, set_point):
        # print(set_point)
        # print("cursor")
        # print(cursor_point)
        d = np.hypot(set_point[0] - cursor_point[0], set_point[1] - cursor_point[1])
        indseq, = np.nonzero(d == d.min())
        ind = indseq[0]

        if d[ind] >= self.epsolon:
            # print(f'rejeito com distancia: {d[ind]}')
            ind = None
        else:
            # print(f'aceito com distancia: {d[ind]}')
            # print(cursor_point)
            print(set_point[0][ind], set_point[1][ind])
            # print("conjunto certo")

        return ind

    def get_ind_under_point(self, event):
        """
        Return the index of the point closest to the event position or *None*
        if no point is within ``self.epsilon`` to the event position.
        """
        # display coords
        src_points_display = self.transform_display_coords(self.landmarks_src, is_tgt=False)
        tgt_points_display = self.transform_display_coords(self.landmarks_tgt, is_tgt=True)

        ind_src = self.check_point_in_set((event.xdata, event.ydata), (src_points_display[0], src_points_display[1]))
        ind_tgt = self.check_point_in_set((event.xdata, event.ydata), (tgt_points_display[0], tgt_points_display[1]))

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
            self.scatter_and_plot(self.landmarks_src, self.landmarks_tgt, self.frame_dims)

        if event.button is MouseButton.MIDDLE:
            # print(event.button)
            if x <= self.frame_dims[1]:
                x, y = self.convert_coord_disp2src(x, y)
                newrow = [y, x]
                if self.landmarks_src is not None and len(self.landmarks_src):
                    self.landmarks_src = np.vstack([self.landmarks_src, newrow])
                else:
                    self.landmarks_src = np.array([newrow])
            else:
                x, y = self.convert_coord_disp2src(x, y, is_tgt=True)
                newrow = [y, x]
                if self.landmarks_tgt is not None and len(self.landmarks_tgt):
                    self.landmarks_tgt = np.vstack([self.landmarks_tgt, newrow])
                else:
                    self.landmarks_tgt = np.array([newrow])

            self.scatter_and_plot(self.landmarks_src, self.landmarks_tgt, self.frame_dims)

    def on_button_release(self, event):
        """Callback for mouse button releases."""
        if event.button != 1:
            return
        self._ind_src, self._ind_tgt = None, None

    def on_key_press(self, event):
        """Callback for key presses."""
        if not event.inaxes:
            return
        x, y = event.xdata, event.ydata

    def convert_coord_disp2src(self, x, y, is_tgt=False):
        # print(self.frame_dims)

        if is_tgt:
            x = x - self.frame_dims[1]

        x_new = x / self.frame_dims[1]
        y_new = y / self.frame_dims[0]

        return 2*x_new - 1, 2*y_new - 1

    def on_mouse_move(self, event):
        """Callback for mouse movements."""
        if event.inaxes is None:
            return

        x, y = event.xdata, event.ydata

        if self._ind_src is not None:
            self.landmarks_src[self._ind_src, 1], self.landmarks_src[self._ind_src,0] = self.convert_coord_disp2src(x, y)

        if self._ind_tgt is not None:
            self.landmarks_tgt[self._ind_tgt, 1], self.landmarks_tgt[self._ind_tgt,0] = self.convert_coord_disp2src(x, y, True)

        self.scatter_and_plot(self.landmarks_src, self.landmarks_tgt, self.frame_dims)

    def return_points(self):
        return (self.landmarks_src, self.landmarks_tgt)


if __name__ == '__main__':
    import os.path as osp

    fileimg1 = osp.join("data", "frll", "001_03.jpg")
    fileimg2 = osp.join("data", "frll", "002_03.jpg")

    img1 = cv2.cvtColor(cv2.imread(fileimg1), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(fileimg2), cv2.COLOR_BGR2RGB)

    img1 = cv2.resize(img1, (520, 520))
    img2 = cv2.resize(img2, (520, 520))

    p = FaceInteractor(img1, img2)
    plt.show()

    src, tgt = p.return_points()

    print(src)
    print(tgt)
