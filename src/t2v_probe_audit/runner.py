"""The three experiments.

regime.json          floor33 (16 frames @ 8 fps) and gen2 (48 frames @ 24 fps)
regime_control.json  gen2 sampled every third frame (16 frames @ 8 fps)
multigen.json        five-generator consequence analysis under I5

Clips are independent and are scored in parallel processes. The partial result
is written after each generator.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import corpus, embedding, pipeline, stats
from .config import (
    CONTROL_JSON, GENERATORS, HIGHER_BETTER, MULTIGEN_JSON, REGIME_JSON,
)

WORKER_STATE = {"model": None, "with_embedding": False}


def _init_worker(with_embedding: bool):
    """One thread per worker process."""
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    WORKER_STATE["with_embedding"] = with_embedding
    WORKER_STATE["model"] = embedding.load_embedder() if with_embedding else None


def _audit_task(task):
    path, seed, ivs, step = task
    return pipeline.audit_clip(WORKER_STATE["model"], path, seed,
                               with_embedding=WORKER_STATE["with_embedding"],
                               ivs=ivs, step=step)


def _default_workers() -> int:
    return min(8, os.cpu_count() or 1)


def _run_generator(generator, n, seed, ivs, step, with_embedding, workers,
                   progress=True, label=None):
    """Audit one sampled generator and return its per-clip records."""
    label = label or generator
    picked = corpus.sample_clips(generator, n, seed)
    tasks = [(p, seed, ivs, step) for p in picked]
    records = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(with_embedding,)) as ex:
        for i, rec in enumerate(ex.map(_audit_task, tasks), 1):
            if rec is not None:
                rec = {k: v for k, v in rec.items()
                       if k in ("clip", "n_frames", "base", *ivs)}
                rec["generator"] = generator
                records.append(rec)
            if progress and i % 10 == 0:
                print(f"[{label}] {i}/{len(picked)}", flush=True)
    return records


def _write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def _coverage_block(records):
    return {
        "disc_defined": sum(1 for r in records
                            if r["base"]["mcs_disc_guard"] is not None),
        "disc_coverage_mean": float(np.mean([r["base"]["coverage"]
                                             for r in records])),
        "composite_defined": sum(1 for r in records
                                 if r["base"]["mcs_guard"] is not None),
        "n": len(records),
    }


def run_regimes(n=60, seed=2026, workers=None,
                ivs=("I1_reverse", "I3_shuffle", "I5_speed0p5x"),
                with_embedding=True, out: Path = REGIME_JSON):
    """Two native regimes, written to one file keyed by generator."""
    workers = workers or _default_workers()
    payload = {"n": n, "seed": seed, "interventions": list(ivs),
               "step": 1, "per_generator": {}}
    for generator in ("floor33", "gen2"):
        records = _run_generator(generator, n, seed, list(ivs), 1,
                                 with_embedding, workers)
        payload["per_generator"][generator] = {
            "coverage": _coverage_block(records),
            "per_clip": records,
        }
        _write(out, payload)
        print(f"[{generator}] saved {len(records)} clips", flush=True)
    print(f"saved: {out}", flush=True)
    return payload


def run_density_control(n=60, seed=2026, workers=None,
                        ivs=("I1_reverse", "I3_shuffle", "I5_speed0p5x"),
                        with_embedding=True, out: Path = CONTROL_JSON,
                        label="gen2_k3"):
    """gen2 kept every third frame: the floor33 density on one generator."""
    workers = workers or _default_workers()
    records = _run_generator("gen2", n, seed, list(ivs), 3,
                             with_embedding, workers, label=label)
    payload = {"n": n, "seed": seed, "interventions": list(ivs),
               "step": 3, "per_generator": {
                   label: {"coverage": _coverage_block(records),
                           "per_clip": records}}}
    _write(out, payload)
    print(f"[{label}] saved {len(records)} clips -> {out}", flush=True)
    return payload


def _report(records, guarded):
    """Mean reported composite and its support under one scoring rule."""
    vals, cov = [], []
    for r in records:
        m = r["base"]
        if guarded:
            v = m["mcs_guard"]
        else:
            disc = 0.5 if m["mcs_disc"] is None else m["mcs_disc"]
            v = (disc + m["mcs_warp"] + m["mcs_mag"]) / 3.0
        if v is not None:
            vals.append(v)
        cov.append(bool(v is not None))
    return (float(np.mean(vals)) if vals else None), float(np.mean(cov))


def run_consequence(n=60, seed=2026, workers=None, out: Path = MULTIGEN_JSON,
                    ivs=("I5_speed0p5x",)):
    """Five-generator consequence analysis, pre-repair against repaired."""
    workers = workers or _default_workers()
    payload = {"n": n, "seed": seed, "interventions": list(ivs), "generators": {}}
    for generator in GENERATORS:
        records = _run_generator(generator, n, seed, list(ivs), 1,
                                 False, workers)
        pre, pre_cov = _report(records, guarded=False)
        post, post_cov = _report(records, guarded=True)

        d_rate = [stats.delta_bad(r["base"], r["I5_speed0p5x"], "mcs",
                                  HIGHER_BETTER) for r in records]
        art = float(np.nanmean([x for x in d_rate if x is not None])) if any(
            x is not None for x in d_rate) else None

        pre_map = {r["clip"]: ((0.5 if r["base"]["mcs_disc"] is None
                                else r["base"]["mcs_disc"])
                               + r["base"]["mcs_warp"]
                               + r["base"]["mcs_mag"]) / 3.0
                   for r in records}
        post_map = {r["clip"]: r["base"]["mcs_guard"] for r in records}
        inv, n_inv, n_pairs = stats.pairwise_inversion_rate(pre_map, post_map)

        payload["generators"][generator] = {
            "reported_pre": pre, "coverage_pre": pre_cov,
            "reported_post": post, "coverage_post": post_cov,
            "artifact_i5": art,
            "coverage_disc_mean": float(np.mean([r["base"]["coverage"]
                                                 for r in records])),
            "n": len(records),
            "pairwise_inversion": inv, "inverted_pairs": n_inv,
            "pairs": n_pairs,
        }
        _write(out, payload)
        print(f"[{generator}] n={len(records)} pre={pre:.4f} post={post:.4f} "
              f"cov={post_cov:.3f} art={art:+.4f} inv={inv}", flush=True)

    pre_rank = {g: payload["generators"][g]["reported_pre"] for g in GENERATORS}
    post_rank = {g: payload["generators"][g]["reported_post"] for g in GENERATORS}
    gaps = [v for v in pre_rank.values() if v is not None]
    max_art = float(max(abs(payload["generators"][g]["artifact_i5"] or 0)
                        for g in GENERATORS))
    spread = (max(gaps) - min(gaps)) if gaps else None
    payload["consequence"] = {
        "ranking_pre": dict(sorted(pre_rank.items(), key=lambda kv: -kv[1])),
        "ranking_post": dict(sorted(post_rank.items(), key=lambda kv: -kv[1])),
        "kendall_tau": stats.kendall_tau(pre_rank, post_rank),
        "between_generator_gap": spread,
        "max_abs_artifact": max_art,
        "artifact_over_gap": (max_art / spread) if spread else None,
    }
    _write(out, payload)
    print(json.dumps(payload["consequence"], indent=1), flush=True)
    return payload
