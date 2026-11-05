"""Unit tests for the audit primitives.

Run with:  python -m pytest tests -q
"""

import numpy as np

from t2v_probe_audit import metrics, probes, stats
from t2v_probe_audit.config import HIGHER_BETTER


def test_all_pairs_mean_is_permutation_invariant():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(12, 8))
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    sim = f @ f.T
    iu = np.triu_indices(len(f), k=1)
    a = float(np.mean(sim[iu]))
    perm = rng.permutation(12)
    sim_p = f[perm] @ f[perm].T
    b = float(np.mean(sim_p[np.triu_indices(len(f), k=1)]))
    assert abs(a - b) < 1e-12


def test_i5_changes_no_pixel_value():
    frames = [np.full((6, 6, 3), v, dtype=np.uint8) for v in (10, 20, 30)]
    dup = probes.interventions(frames, 0)["I5_speed0p5x"]
    assert len(dup) == 2 * len(frames)
    assert set(map(id, dup)) <= set(map(id, frames))
    assert np.array_equal(np.unique(np.stack(dup)), np.array([10, 20, 30], dtype=np.uint8))


def test_guard_rejects_duplicated_frames():
    frame = np.random.default_rng(1).uniform(0, 1, (32, 32)).astype(np.float32)
    zero = np.zeros((32, 32, 2), dtype=np.float32)
    residual = metrics.fit_global(zero)
    guarded = metrics.guard_residual(zero, residual)
    assert not metrics.has_motion(guarded)
    assert metrics.clip_metrics([frame, frame.copy()])["mcs_disc"] == 0.5


def test_specificity_ratio_and_dominance():
    order = [0.10, 0.20, 0.05, 0.15]
    rate = [-0.30, -0.05, -0.01, -0.18]
    ratio = stats.specificity_ratio(float(np.mean(order)), float(np.mean(rate)))
    assert ratio > 1.0
    dom, n = stats.dominance(order, rate)
    assert n == 4 and dom == 0.5


def test_badness_signs():
    # A drop in a higher-is-better metric is a positive badness delta;
    # a rise in an error metric is also positive.
    assert abs(stats.delta_bad({"a": 0.9}, {"a": 0.8}, "a", {"a"}) - 0.1) < 1e-9
    assert abs(stats.delta_bad({"a": 0.1}, {"a": 0.2}, "a", set()) - 0.1) < 1e-9


def test_composite_is_none_when_a_channel_is_undefined():
    m = {"mcs_disc": None, "mcs_warp": 0.5, "mcs_mag": 0.5,
         "mcs_disc_guard": None, "mcs_warp_guard": 0.5, "mcs_mag_guard": 0.5}
    assert metrics.composite(m, guarded=False) is None
    assert metrics.composite(m, guarded=True) is None


def test_probe_specs_match_the_implemented_suite():
    keys = [p["key"] for p in probes.PROBE_SPECS]
    frames = [np.full((8, 8, 3), v, dtype=np.uint8) for v in range(8)]
    implemented = set(probes.interventions(frames, 0))
    assert set(keys) == implemented
    assert {"I3_shuffle", "I5_speed0p5x"} <= set(keys)
    assert sum(1 for p in probes.PROBE_SPECS if p["reported"]) == 3
