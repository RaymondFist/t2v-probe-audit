"""The six interventions of the probe suite.

I1 reverse        full time reversal
I2 window reverse reverse within windows of four frames
I3 shuffle        deterministic permutation of frames (order destroyed)
I4 speed 2x       keep every second frame (displacement up)
I5 speed 0.5x     duplicate every frame (displacement down)
I6 block jitter   cyclic 16-px shifts of the highest-variance 3x3 cell

I3 permutes the frame order and I5 duplicates every frame. Neither introduces a
new pixel value and neither re-encodes the clip.
"""

import cv2
import numpy as np

# Metadata for Table 1. Keys are the columns of tables/table1_probe_suite.csv.
PROBE_SPECS = [
    {
        "id": "I1",
        "key": "I1_reverse",
        "name": "reverse",
        "operation": "reverse the whole clip",
        "pixel_values": "unchanged",
        "frame_order": "reversed",
        "adjacent_pairs": "same multiset",
        "role": "anchor diagnostic",
        "reported": True,
    },
    {
        "id": "I2",
        "key": "I2_window_rev",
        "name": "window reverse",
        "operation": "reverse inside windows of 4",
        "pixel_values": "unchanged",
        "frame_order": "partly reversed",
        "adjacent_pairs": "mostly same",
        "role": "anchor diagnostic",
        "reported": False,
    },
    {
        "id": "I3",
        "key": "I3_shuffle",
        "name": "shuffle",
        "operation": "seeded permutation",
        "pixel_values": "unchanged",
        "frame_order": "destroyed",
        "adjacent_pairs": "changed",
        "role": "order sensitivity",
        "reported": True,
    },
    {
        "id": "I4",
        "key": "I4_speed2x",
        "name": "2x",
        "operation": "drop every other frame",
        "pixel_values": "subset",
        "frame_order": "kept",
        "adjacent_pairs": "subset",
        "role": "rate, faster",
        "reported": False,
    },
    {
        "id": "I5",
        "key": "I5_speed0p5x",
        "name": "0.5x",
        "operation": "duplicate every frame",
        "pixel_values": "unchanged",
        "frame_order": "kept",
        "adjacent_pairs": "duplicated",
        "role": "rate, slower",
        "reported": True,
    },
    {
        "id": "I6",
        "key": "I6_obj_jitter",
        "name": "block jitter",
        "operation": "shift highest-variance cell",
        "pixel_values": "locally changed",
        "frame_order": "kept",
        "adjacent_pairs": "changed",
        "role": "local inconsistency",
        "reported": False,
    },
]


def window_reverse(frames, w: int = 4):
    out = []
    for i in range(0, len(frames), w):
        out.extend(frames[i:i + w][::-1])
    return out


def block_jitter(frames, amp: int = 4, cells: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    h, w = frames[0].shape[:2]
    ch, cw = h // cells, w // cells
    gray_stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
                           for f in frames])
    var = gray_stack.var(axis=0)
    best, best_v = (0, 0), -1.0
    for cy in range(cells):
        for cx in range(cells):
            v = float(var[cy * ch:(cy + 1) * ch, cx * cw:(cx + 1) * cw].mean())
            if v > best_v:
                best, best_v = (cy, cx), v
    cy, cx = best
    y0, x0 = cy * ch, cx * cw
    out = []
    for f in frames:
        dx, dy = rng.integers(-amp, amp + 1, size=2)
        canvas = f.copy()
        canvas[y0:y0 + ch, x0:x0 + cw] = np.roll(
            f[y0:y0 + ch, x0:x0 + cw], (dy, dx), axis=(0, 1))
        out.append(canvas)
    return out


def interventions(frames, seed: int):
    """All six variants of one decoded clip, derived deterministically."""
    rng = np.random.default_rng(seed)
    n = len(frames)
    return {
        "I1_reverse": frames[::-1],
        "I2_window_rev": window_reverse(frames, 4),
        "I3_shuffle": [frames[i] for i in rng.permutation(n)],
        "I4_speed2x": frames[::2],
        "I5_speed0p5x": [f for f in frames for _ in (0, 1)],
        "I6_obj_jitter": block_jitter(frames, seed=seed),
    }
