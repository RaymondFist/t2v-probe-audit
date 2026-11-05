"""End-to-end audit of one clip under a subset of the probe suite."""

from . import corpus, embedding, metrics, probes


def audit_clip(model, path, seed: int = 2026, with_embedding: bool = True,
               ivs=None, step: int = 1):
    """Return a record with base metrics and per-intervention metrics.

    ivs restricts which interventions are applied, so callers that need only the
    matched pair do not pay for the rest of the suite. step > 1 keeps every
    step-th frame after the 2.0 s horizon is applied; this is the within-generator
    density control, and it never touches any pixel value.
    """
    n = corpus.horizon_frames(path)
    bgr, rgb = corpus.read_clip(path, max_frames=n)
    if step > 1:
        bgr = bgr[::step]
        rgb = corpus.rgb_for_frames(bgr)
    if len(bgr) < 4:
        return None

    base = metrics.clip_metrics(metrics.to_gray(bgr))
    if base is None:
        return None
    base["mcs"] = metrics.composite(base, guarded=False)
    base["mcs_guard"] = metrics.composite(base, guarded=True)
    if with_embedding:
        base.update(embedding.embedding_stats(model, rgb))

    rec = {"clip": path.name, "n_frames": len(bgr), "base": base}

    all_ivs = probes.interventions(bgr, seed)
    if ivs is not None:
        all_ivs = {k: v for k, v in all_ivs.items() if k in ivs}
    for name, frames in all_ivs.items():
        m = metrics.clip_metrics(metrics.to_gray(frames))
        if m is None:
            continue
        m["mcs"] = metrics.composite(m, guarded=False)
        m["mcs_guard"] = metrics.composite(m, guarded=True)
        if with_embedding:
            m.update(embedding.embedding_stats(model, corpus.rgb_for_frames(frames)))
        rec[name] = m
    return rec
