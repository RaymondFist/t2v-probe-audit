"""Pixel-level metrics: working-resolution conversion and local SSIM."""

import cv2
import numpy as np

from ..config import WORK_H, WORK_W


def to_gray(frames_bgr):
    """Convert BGR frames to working-resolution float grayscale in [0, 1]."""
    out = []
    for frame in frames_bgr:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape[1] != WORK_W:
            gray = cv2.resize(gray, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        out.append(gray.astype(np.float32) / 255.0)
    return out


def ssim_single(a, b):
    """Local SSIM on float images in [0, 1] (11x11 Gaussian, sigma 1.5)."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sig_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
    sig_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
    sig_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab
    ssim_map = ((2 * mu_ab + c1) * (2 * sig_ab + c2)) / (
        (mu_a2 + mu_b2 + c1) * (sig_a2 + sig_b2 + c2))
    return float(np.mean(ssim_map))
