"""Statistics for the audit: badness deltas, specificity ratios, support."""

import numpy as np


def delta_bad(base, variant, metric, higher_better):
    """Signed change in badness; a positive value indicates a worse score."""
    def bad(m, v):
        return -v if m in higher_better else v
    if base.get(metric) is None or variant.get(metric) is None:
        return None
    return bad(metric, variant[metric]) - bad(metric, base[metric])


def summarize(values):
    """Mean, median, std, effect size and sign share of a paired delta list."""
    v = np.array([x for x in values if x is not None], dtype=np.float64)
    if v.size == 0:
        return None
    out = {
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "n": int(v.size),
        "frac_positive": float((v > 0).mean()),
        "frac_negative": float((v < 0).mean()),
    }
    out["dz"] = float(v.mean() / (v.std(ddof=1) + 1e-12)) if v.size > 1 else 0.0
    return out


def specificity_ratio(d_order, d_rate, eps: float = 1e-9):
    """|mean rate effect| / |mean order effect|."""
    if d_order is None or d_rate is None:
        return None
    return float(abs(d_rate) / (abs(d_order) + eps))


def dominance(d_order, d_rate):
    """Fraction of clips on which the rate effect exceeds the order effect."""
    pairs = list(zip(d_order, d_rate))
    n = sum(1 for a, b in pairs if a is not None and b is not None)
    if n == 0:
        return None, 0
    k = sum(1 for a, b in pairs if a is not None and b is not None and abs(b) > abs(a))
    return float(k / n), n


def bootstrap_ci(values, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    v = np.array([x for x in values if x is not None], dtype=np.float64)
    if v.size < 2:
        return None
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(v, size=v.size, replace=True).mean()
                      for _ in range(n_boot)])
    return [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


def kendall_tau(order_a, order_b):
    """Kendall tau between two rankings given as dicts {name: value}."""
    names = list(order_a)
    concordant = discordant = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = order_a[names[i]] - order_a[names[j]]
            b = order_b[names[i]] - order_b[names[j]]
            if a * b > 0:
                concordant += 1
            elif a * b < 0:
                discordant += 1
    total = concordant + discordant
    return float((concordant - discordant) / total) if total else None


def pairwise_inversion_rate(pre, post):
    """Fraction of clip pairs whose ordering differs between two score maps."""
    names = [k for k in pre if pre[k] is not None and post.get(k) is not None]
    inv = total = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = pre[names[i]] - pre[names[j]]
            b = post[names[i]] - post[names[j]]
            total += 1
            if a * b < 0:
                inv += 1
    return (float(inv / total) if total else None), inv, total
