"""Corpus provenance record written to data/manifest.json.

One entry per generator with clip count, observed frame rate, frame-count
range, resolution, byte size and the sorted file list used by the sampler.
Every field is read from the container headers on disk.
"""

import json
from datetime import datetime, timezone

from . import corpus
from .config import HORIZON_SECONDS, MANIFEST, RAW_DIR

DATASET = {
    "name": "EvalCrafter ECTV text-to-video video release",
    "benchmark": "EvalCrafter (Liu et al., CVPR 2024)",
    "huggingface_repo": "RaphaelLiu/EvalCrafter_T2V_Dataset",
    "note": ("Public video set distributed with the EvalCrafter benchmark. "
             "Clips are third-party material and are not redistributed with this "
             "repository; only this record of the local copy is stored."),
}


def _generator_block(generator):
    clips = corpus.list_clips(generator)
    headers = [corpus.read_header(p) for p in clips]
    fps = sorted({round(h["fps"], 3) for h in headers})
    frames = [h["frames"] for h in headers]
    sizes = sorted({(h["width"], h["height"]) for h in headers})
    total_bytes = sum(p.stat().st_size for p in clips)
    return {
        "clips": len(clips),
        "fps_observed": fps,
        "frames_min": min(frames) if frames else None,
        "frames_max": max(frames) if frames else None,
        "resolutions": [f"{w}x{h}" for w, h in sizes],
        "total_bytes": total_bytes,
        "horizon_frames_at_native_fps": (
            corpus.horizon_frames(clips[0]) if clips else None),
        "files": [p.name for p in clips],
    }


def build_manifest(force: bool = False, out=MANIFEST):
    """Write the corpus record; skip when present unless force is set."""
    if out.exists() and not force:
        print(f"manifest exists: {out} (use --force to rebuild)")
        return json.loads(out.read_text(encoding="utf-8"))

    generators = {}
    for generator in sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir()):
        block = _generator_block(generator)
        generators[generator] = block
        print(f"[manifest] {generator}: {block['clips']} clips, "
              f"fps {block['fps_observed']}", flush=True)

    payload = {
        "dataset": DATASET,
        "horizon_seconds": HORIZON_SECONDS,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generators": generators,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return payload
