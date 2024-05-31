#!/usr/bin/env python
# coding: utf-8

import os
import os.path as osp
from multiprocessing import Pool
import sys
import dlib
import numpy as np
import PIL.Image
import scipy.ndimage
from argparse import ArgumentParser


def image_align(src_file,
                dst_file,
                face_landmarks,
                output_size=1024,
                transform_size=4096,
                enable_padding=True,
                just_crop=False):
    # Align function from FFHQ dataset pre-processing step
    # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    lm = np.array(face_landmarks)
    # lm_chin = lm[0:17]  # left-right
    # lm_eyebrow_left = lm[17:22]  # left-right
    # lm_eyebrow_right = lm[22:27]  # left-right
    # lm_nose = lm[27:31]  # top-down
    # lm_nostrils = lm[31:36]  # top-down
    lm_eye_left = lm[36:42]  # left-clockwise
    lm_eye_right = lm[42:48]  # left-clockwise
    lm_mouth_outer = lm[48:60]  # left-clockwise
    # lm_mouth_inner = lm[60:68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    if just_crop:
        x = np.array([1, 0], dtype=np.float64)
    else:
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Load in-the-wild image.
    if not os.path.isfile(src_file):
        print(
            '\nCannot find source image. Please run "--wilds" before "--align".'
        )
        return
    img = PIL.Image.open(src_file)
    img = img.convert('RGB')

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)),
                 int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.LANCZOS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
            min(crop[2] + border,
                img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
           int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border,
               0), max(-pad[1] + border,
                       0), max(pad[2] - img.size[0] + border,
                               0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img),
                     ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 -
            np.minimum(np.float32(x) / pad[0],
                       np.float32(w - 1 - x) / pad[2]), 1.0 -
            np.minimum(np.float32(y) / pad[1],
                       np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) -
                img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)),
                                  'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform(
        (transform_size, transform_size),
        PIL.Image.Transform.QUAD,
        (quad + 0.5).flatten(), PIL.Image.Resampling.BILINEAR
    )
    if output_size < transform_size:
        img = img.resize((output_size, output_size),
                         PIL.Image.Resampling.LANCZOS)

    # Save aligned image.
    img.save(dst_file, 'PNG')


class LandmarksDetector:
    def __init__(self, predictor_model_path):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        self.detector = dlib.get_frontal_face_detector(
        )  # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image):
        img = dlib.load_rgb_image(image)
        dets = self.detector(img, 1)

        for detection in dets:
            face_landmarks = [
                (item.x, item.y)
                for item in self.shape_predictor(img, detection).parts()
            ]
            yield face_landmarks


def work_landmark(raw_img_path, img_name, face_landmarks, output_size=256,
                  just_crop=False):
    face_img_name = '%s.png' % (os.path.splitext(img_name)[0], )
    aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
    if os.path.exists(aligned_face_path):
        return
    image_align(raw_img_path,
                aligned_face_path,
                face_landmarks,
                output_size=output_size,
                just_crop=just_crop)


if __name__ == "__main__":
    """
    Extracts and aligns all faces from images using DLib and a function from
    original FFHQ dataset preparation step.
    Example run: python align_images.py /raw_images /aligned_images
    """
    DEFAULT_LM_PATH = "landmark_models/shape_predictor_68_face_landmarks_GTX.dat"
    parser = ArgumentParser()
    parser.add_argument(
        "input_imgs_path", type=str, help="input images directory path"
    )
    parser.add_argument(
        "output_imgs_path", type=str, help="output images directory path"
    )
    parser.add_argument(
        "--landmark-detector", type=str, default=DEFAULT_LM_PATH,
        help=f"Path to the landmark detector model. By default we assume"
        f" it as \"{DEFAULT_LM_PATH}\".")
    parser.add_argument(
        "--just-crop", "-c", action="store_true", help="Perform no image"
        " alignment, just crop the face."
    )
    parser.add_argument(
        "--output-size", type=int, default=256, help="Size of the output"
        " images. By default is 256, meaning 256x256 pixels."
    )
    parser.add_argument(
        "--n-tasks", type=int, default=4, help="Number of parallel jobs to"
        " run. By default is 4."
    )
    args = parser.parse_args()

    if not osp.exists(args.landmark_detector):
        print(
            "[ERROR] Landmark detector not found. Download it first and store"
            f" it at \"{args.landmark_detector}\". See the \"Makefile\" for a"
            " rule to do so automatically.",
            file=sys.stderr
        )
        sys.exit(1)

    RAW_IMAGES_DIR = args.input_imgs_path
    ALIGNED_IMAGES_DIR = args.output_imgs_path

    if not osp.exists(ALIGNED_IMAGES_DIR):
        os.makedirs(ALIGNED_IMAGES_DIR)

    files = sorted([f for f in os.listdir(RAW_IMAGES_DIR) if f.endswith("png") or f.endswith("jpg")])
    print(f"Total image files: {len(files)}")

    def err_cb(e):
        print(f"error: {e}", file=sys.stderr)

    with Pool(args.n_tasks) as pool:
        res = []
        landmarks_detector = LandmarksDetector(args.landmark_detector)
        for img_name in files:
            raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
            for i, face_landmarks in enumerate(
                    landmarks_detector.get_landmarks(raw_img_path),
                    start=1):

                job = pool.apply_async(
                    work_landmark,
                    (raw_img_path, img_name, face_landmarks, args.output_size, args.just_crop),
                    error_callback=err_cb
                )
                res.append(job)

        pool.close()
        pool.join()

    print(f"Output aligned images at: {ALIGNED_IMAGES_DIR}")
