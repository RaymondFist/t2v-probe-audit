"""Dense flow, flow warping and the partial-affine global-motion fit."""

import cv2
import numpy as np


def warp_with_flow(img, flow):
    """Warp img by a dense flow field with a replicated border."""
    h, w = img.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = (xs + flow[:, :, 0]).astype(np.float32)
    map_y = (ys + flow[:, :, 1]).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def fit_global(flow):
    """Residual after a robust partial-affine global-motion fit.

    The fit is computed on an 8-px grid and inpainted back to full resolution.
    """
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[8:h:8, 8:w:8]
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    f = flow[ys, xs].reshape(-1, 2)
    dst = pts + f
    cv2.setRNGSeed(2026)  # RANSAC draws from OpenCV's global RNG
    model, _ = cv2.estimateAffinePartial2D(
        pts, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if model is None:
        return flow.copy()
    a = model[:, :2]
    b = model[:, 2]
    induced = (a @ pts.T).T + b - pts
    residual = f - induced
    res = np.zeros_like(flow)
    res[ys.ravel(), xs.ravel()] = residual
    for c in range(2):
        field = res[:, :, c].copy()
        mask = np.zeros_like(field, dtype=bool)
        mask[ys.ravel(), xs.ravel()] = True
        field = cv2.inpaint(field.astype(np.float32),
                            (~mask).astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
        res[:, :, c] = field
    return res
