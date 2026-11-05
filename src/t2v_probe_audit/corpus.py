"""Clip listing, sampling, decoding and truncation.

Interventions are applied to decoded arrays in memory; no clip is re-encoded.
Sampling sorts before selecting, which makes the sample independent of
filesystem enumeration order.
"""

from pathlib import Path

import cv2
import numpy as np

from .config import DINO_SIZE, HORIZON_SECONDS, RAW_DIR


def list_clips(generator: str, data_root: Path = RAW_DIR) -> list:
    """All clips of one generator, in a platform-independent order."""
    return sorted((Path(data_root) / generator).glob("*.mp4"))


def sample_clips(generator: str, n: int, seed: int = 2026,
                 data_root: Path = RAW_DIR) -> list:
    """Deterministic sample: sort, then pick indices with a seeded RNG."""
    clips = list_clips(generator, data_root)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(clips), min(n, len(clips)), replace=False))
    return [clips[i] for i in idx]


def read_clip(path, max_frames=None, dino_size: int = DINO_SIZE):
    """Return (bgr_frames, rgb_float_array).

    bgr_frames keep native resolution. rgb is resized to dino_size and converted
    to RGB in [0, 1].
    """
    cap = cv2.VideoCapture(str(path))
    bgr, rgb = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        bgr.append(frame)
        rgb.append(cv2.cvtColor(
            cv2.resize(frame, (dino_size, dino_size), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB))
    cap.release()
    if max_frames is not None:
        bgr = bgr[:max_frames]
        rgb = rgb[:max_frames]
    return bgr, np.stack(rgb).astype(np.float32) / 255.0


def native_fps(path) -> float:
    """Frame rate recorded in the container, not a default."""
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return fps


def horizon_frames(path, seconds: float = HORIZON_SECONDS) -> int:
    """Frame count spanned by the time horizon at the clip's native rate."""
    fps = native_fps(path) or 8.0
    return max(2, int(round(fps * seconds)))


def read_header(path) -> dict:
    """Container metadata without decoding pixels; used to build the manifest."""
    cap = cv2.VideoCapture(str(path))
    info = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def rgb_for_frames(bgr_frames, dino_size: int = DINO_SIZE):
    """Embedding input for an intervened sequence: DINO-size RGB in [0, 1]."""
    return np.stack([
        cv2.cvtColor(cv2.resize(f, (dino_size, dino_size), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2RGB)
        for f in bgr_frames
    ]).astype(np.float32) / 255.0
