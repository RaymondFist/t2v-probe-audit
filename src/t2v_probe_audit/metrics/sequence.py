"""Per-sequence metric assembly, raw and guarded.

The raw and guarded channels are computed from the same flow field.
"""

import cv2
import numpy as np

from ..config import FLOW_PARAMS, LAMBDA_MAG, PAIR_MIN_MAG
from .flow import fit_global, warp_with_flow
from .guard import guard_residual, has_motion
from .pixel import ssim_single


def direction_consistency(residual, prev_residual, thr: float = PAIR_MIN_MAG):
    """Per-pixel direction agreement in [0,1]; 0.5 when there is nothing to measure."""
    dot = residual[:, :, 0] * prev_residual[:, :, 0] + residual[:, :, 1] * prev_residual[:, :, 1]
    denom = (np.sqrt(residual[:, :, 0] ** 2 + residual[:, :, 1] ** 2)
             * np.sqrt(prev_residual[:, :, 0] ** 2 + prev_residual[:, :, 1] ** 2))
    valid = denom > thr * thr
    if int(valid.sum()) < 50:
        return 0.5
    cos = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
    return float(np.mean(np.arccos(cos)) / np.pi)


def pair_angle(residual, prev_residual, thr: float = PAIR_MIN_MAG):
    """Direction disagreement for the guarded chain; None when undefined."""
    dot = residual[:, :, 0] * prev_residual[:, :, 0] + residual[:, :, 1] * prev_residual[:, :, 1]
    denom = (np.sqrt(residual[:, :, 0] ** 2 + residual[:, :, 1] ** 2)
             * np.sqrt(prev_residual[:, :, 0] ** 2 + prev_residual[:, :, 1] ** 2))
    valid = denom > thr * thr
    if int(valid.sum()) < 50:
        return None
    cos = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
    return float(np.mean(np.arccos(cos)) / np.pi)


def clip_metrics(gray):
    """All flow/pixel metrics, raw and guarded, for one frame sequence."""
    T = len(gray)
    if T < 2:
        return None

    mafd, ssim_inst, warp_err, comp_raw, comp_g = [], [], [], [], []
    disc_raw, disc_g, mags_raw, mags_g = [], [], [], []
    prev_raw = prev_g = None
    discarded = 0

    for t in range(T - 1):
        a, b = gray[t], gray[t + 1]
        h, w = a.shape
        identical = np.array_equal(a, b)
        if identical:
            # Duplicated frame: every channel has a closed form and the flow
            # fit is skipped.
            mafd.append(0.0)
            ssim_inst.append(0.0)
            warp_err.append(0.0)
            comp_raw.append(1.0)
            zero = np.zeros((h, w, 2), dtype=np.float32)
            mags_raw.append(zero[:, :, 0])
            mags_g.append(zero[:, :, 0])
            if prev_raw is not None:
                disc_raw.append(direction_consistency(zero, prev_raw))
            prev_raw = zero
            if prev_g is not None:
                ang = pair_angle(zero, prev_g)
                if ang is not None:
                    disc_g.append(ang)
            discarded += 1
            continue

        mafd.append(float(np.mean(np.abs(a - b))))
        ssim_inst.append(1.0 - ssim_single(a, b))

        fa = (a * 255).astype(np.uint8)
        fb = (b * 255).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(fa, fb, None, **FLOW_PARAMS)

        residual = fit_global(flow)
        guarded = guard_residual(flow, residual)

        warp_err.append(1.0 - ssim_single(warp_with_flow(a, flow), b))
        comp_raw.append(ssim_single(warp_with_flow(a, residual), b))
        mags_raw.append(np.sqrt(residual[:, :, 0] ** 2 + residual[:, :, 1] ** 2))

        if prev_raw is not None:
            disc_raw.append(direction_consistency(residual, prev_raw))
        prev_raw = residual

        # Guarded chain.
        if prev_g is not None:
            ang = pair_angle(guarded, prev_g)
            if ang is not None:
                disc_g.append(ang)
        if has_motion(guarded):
            comp_g.append(ssim_single(warp_with_flow(a, guarded), b))
            prev_g = guarded
        else:
            discarded += 1
        mags_g.append(np.sqrt(guarded[:, :, 0] ** 2 + guarded[:, :, 1] ** 2))

    all_raw = np.concatenate([m.ravel() for m in mags_raw])
    all_guarded = np.concatenate([m.ravel() for m in mags_g]) if mags_g else np.zeros(1)

    return {
        "mafd": float(np.mean(mafd)),
        "ssim_instability": float(np.mean(ssim_inst)),
        "warp_error": float(np.mean(warp_err)),
        "mcs_disc": 1.0 - float(np.mean(disc_raw)) if disc_raw else 0.5,
        "mcs_warp": float(np.mean(comp_raw)),
        "mcs_mag": 1.0 - float(np.tanh(LAMBDA_MAG * float(np.std(all_raw)))),
        "mcs_disc_guard": 1.0 - float(np.mean(disc_g)) if disc_g else None,
        "mcs_warp_guard": float(np.mean(comp_g)) if comp_g else None,
        "mcs_mag_guard": 1.0 - float(np.tanh(LAMBDA_MAG * float(np.std(all_guarded)))),
        "coverage": float(len(disc_g) / max(1, T - 2)),
        "pairs_discarded": int(discarded),
        "n_frames": int(T),
    }


def composite(metrics, guarded=False):
    """Equal-weight composite of the three channels, or None if incomplete."""
    if not guarded:
        parts = [metrics["mcs_disc"], metrics["mcs_warp"], metrics["mcs_mag"]]
    else:
        parts = [metrics["mcs_disc_guard"], metrics["mcs_warp_guard"],
                 metrics["mcs_mag_guard"]]
    if any(p is None for p in parts):
        return None
    return float(sum(parts) / 3.0)
