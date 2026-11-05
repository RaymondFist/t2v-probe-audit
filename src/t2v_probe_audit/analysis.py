"""Numbers and invariant checks derived from the per-clip records.

Reads output/metrics/regime.json and output/metrics/regime_control.json and
writes

    output/analysis/numbers.json   the statistics reported in the manuscript
    output/analysis/checks.json    pass/fail of each invariant
"""

import json
from pathlib import Path

import numpy as np

from . import stats
from .config import (
    CHECKS_JSON, CONTROL_JSON, HIGHER_BETTER, NUMBERS_JSON, PROJECT_ROOT,
    REGIME_JSON,
)

# The ten audited metrics.
ORDER = [
    ("mafd", "MAFD"),
    ("ssim_instability", "SSIM instability"),
    ("warp_error", "Warp error"),
    ("mcs_disc", "MCS direction"),
    ("mcs_warp", "MCS warp"),
    ("mcs_mag", "MCS magnitude"),
    ("mcs", "MCS composite"),
    ("dino_pair_mean", "DINOv2 mean"),
    ("dino_pair_min", "DINOv2 min"),
    ("dino_pair_std", "DINOv2 std"),
]

# Channels whose guarded form is reported.
GUARDED = [("mcs_disc_guard", "MCS direction"), ("mcs_warp_guard", "MCS warp"),
           ("mcs_mag_guard", "MCS magnitude"), ("mcs_guard", "MCS composite")]


def load(path: Path):
    if not path.exists():
        raise SystemExit(f"missing {path}; run the experiments first")
    return json.loads(path.read_text(encoding="utf-8"))


def one_generator(records):
    """All statistics for one regime."""
    out = {"n": len(records), "metrics": {}, "guarded": {}}
    for key, label in ORDER:
        d3 = [stats.delta_bad(r["base"], r["I3_shuffle"], key, HIGHER_BETTER)
              for r in records]
        d5 = [stats.delta_bad(r["base"], r["I5_speed0p5x"], key, HIGHER_BETTER)
              for r in records]
        s3 = stats.summarize(d3)
        s5 = stats.summarize(d5)
        out["metrics"][key] = {
            "label": label,
            "i3": s3["mean"] if s3 else None,
            "i3_ci95": stats.bootstrap_ci(d3, 5000, 0.05, 2026),
            "i3_frac_pos": s3["frac_positive"] if s3 else None,
            "i5": s5["mean"] if s5 else None,
            "i5_ci95": stats.bootstrap_ci(d5, 5000, 0.05, 2026),
            "i5_frac_pos": s5["frac_positive"] if s5 else None,
            "ratio": stats.specificity_ratio(s3["mean"] if s3 else None,
                                             s5["mean"] if s5 else None),
            "dominance": stats.dominance(d3, d5)[0],
            "n3": s3["n"] if s3 else 0,
            "n5": s5["n"] if s5 else 0,
        }
    for key, label in GUARDED:
        d3 = [stats.delta_bad(r["base"], r["I3_shuffle"], key, HIGHER_BETTER)
              for r in records]
        d5 = [stats.delta_bad(r["base"], r["I5_speed0p5x"], key, HIGHER_BETTER)
              for r in records]
        s3 = stats.summarize(d3)
        s5 = stats.summarize(d5)
        out["guarded"][key] = {
            "label": label,
            "i3": s3["mean"] if s3 else None,
            "i3_ci95": stats.bootstrap_ci(d3, 5000, 0.05, 2026),
            "i5": s5["mean"] if s5 else None,
            "n3": s3["n"] if s3 else 0,
            "n5": s5["n"] if s5 else 0,
        }
    out["coverage"] = {
        "disc_coverage_mean": float(np.mean([r["base"]["coverage"] for r in records])),
        "disc_defined": sum(1 for r in records
                            if r["base"]["mcs_disc_guard"] is not None),
        "composite_defined": sum(1 for r in records
                                 if r["base"]["mcs_guard"] is not None),
    }
    # Exact permutation invariance of the all-pairs statistic.
    diffs = [abs(r["I3_shuffle"]["dino_allpairs_mean"] - r["base"]["dino_allpairs_mean"])
             for r in records if "dino_allpairs_mean" in r["base"]]
    out["allpairs_max_abs_delta"] = float(max(diffs)) if diffs else None
    out["allpairs_invariant_count"] = int(sum(1 for d in diffs if d < 1e-12))
    out["allpairs_n"] = len(diffs)
    # Exact zero of the guarded direction channel under 0.5x playback.
    gd = [abs(stats.delta_bad(r["base"], r["I5_speed0p5x"], "mcs_disc_guard",
                              HIGHER_BETTER))
          for r in records
          if r["base"]["mcs_disc_guard"] is not None
          and r["I5_speed0p5x"]["mcs_disc_guard"] is not None]
    out["guard_i5_max_abs_delta"] = float(max(gd)) if gd else None
    out["guard_i5_n"] = len(gd)
    # The adjacent-frame minimum is unchanged by duplication on every clip.
    dm = [stats.delta_bad(r["base"], r["I5_speed0p5x"], "dino_pair_min",
                          HIGHER_BETTER) for r in records]
    dm = [abs(x) for x in dm if x is not None]
    out["dino_min_i5_max_abs_delta"] = float(max(dm)) if dm else None
    # Symmetric pair statistics are unchanged by time reversal.
    sym = [abs(stats.delta_bad(r["base"], r["I1_reverse"], m, HIGHER_BETTER))
           for m in ("mafd", "ssim_instability", "dino_pair_mean", "dino_pair_min")
           for r in records if "I1_reverse" in r]
    out["i1_max_abs_delta"] = float(max(sym)) if sym else None
    out["i1_n"] = len(records)
    return out


def _checks(numbers):
    checks = {}
    for gen, block in numbers["per_regime"].items():
        checks[f"{gen}_allpairs_exactly_invariant"] = {
            "max_abs_delta": block["allpairs_max_abs_delta"],
            "identical_clips": block["allpairs_invariant_count"],
            "n": block["allpairs_n"],
            "pass": block["allpairs_max_abs_delta"] is not None
                    and block["allpairs_max_abs_delta"] < 1e-12,
        }
        checks[f"{gen}_guard_removes_rate_response"] = {
            "max_abs_delta": block["guard_i5_max_abs_delta"],
            "n": block["guard_i5_n"],
            "pass": block["guard_i5_max_abs_delta"] is not None
                    and block["guard_i5_max_abs_delta"] < 1e-12,
        }
        checks[f"{gen}_dino_min_ignores_duplication"] = {
            "max_abs_delta": block["dino_min_i5_max_abs_delta"],
            "pass": block["dino_min_i5_max_abs_delta"] is not None
                    and block["dino_min_i5_max_abs_delta"] < 1e-12,
        }
        checks[f"{gen}_reversal_invariant_pair_statistics"] = {
            "max_abs_delta": block["i1_max_abs_delta"],
            "n": block["i1_n"],
            "pass": block["i1_max_abs_delta"] is not None
                    and block["i1_max_abs_delta"] < 1e-9,
        }
        sensitive = ["mafd", "ssim_instability", "warp_error", "mcs_disc",
                     "mcs_warp", "mcs_mag", "mcs", "dino_pair_mean"]
        checks[f"{gen}_order_sensitivity_positive"] = {
            "metrics": sensitive,
            "pass": all(block["metrics"][m]["i3"] is not None
                        and block["metrics"][m]["i3"] > 0 for m in sensitive),
        }
    return checks


def build_numbers():
    """Recompute numbers.json and checks.json from the per-clip records."""
    regime = load(REGIME_JSON)
    control = load(CONTROL_JSON)

    numbers = {"per_regime": {}, "checks": {}, "provenance": {
        "n": regime["n"], "seed": regime["seed"],
        "interventions": regime["interventions"],
        "density_control_step": control["step"],
        "source": [
            REGIME_JSON.relative_to(PROJECT_ROOT).as_posix(),
            CONTROL_JSON.relative_to(PROJECT_ROOT).as_posix(),
        ],
    }}
    for gen in ("floor33", "gen2"):
        numbers["per_regime"][gen] = one_generator(
            regime["per_generator"][gen]["per_clip"])
    control_key = next(iter(control["per_generator"]))
    numbers["per_regime"][control_key] = one_generator(
        control["per_generator"][control_key]["per_clip"])

    numbers["checks"] = _checks(numbers)

    NUMBERS_JSON.parent.mkdir(parents=True, exist_ok=True)
    NUMBERS_JSON.write_text(json.dumps(numbers, indent=1), encoding="utf-8")
    CHECKS_JSON.write_text(json.dumps(numbers["checks"], indent=1), encoding="utf-8")
    return numbers


def _num(value, spec, width=9):
    return format(value, spec) if value is not None else "n/a".rjust(width)


def report(numbers) -> int:
    """Print the checks and the metric table; return a process exit code."""
    for name, c in numbers["checks"].items():
        print(("PASS " if c["pass"] else "FAIL ") + name, c)

    for gen, block in numbers["per_regime"].items():
        print(f"\n[{gen}] n={block['n']} coverage={block['coverage']['disc_coverage_mean']:.3f}"
              f" defined={block['coverage']['disc_defined']}")
        print(f"{'metric':18s} {'I3':>9s} {'I5':>9s} {'R':>6s} {'D':>6s}"
              f" {'gI3':>9s} {'gI5':>9s}")
        for key, label in ORDER:
            m = block["metrics"][key]
            g = block["guarded"].get(key + "_guard", {})
            print(f"{label:18s} {_num(m['i3'], '+9.4f')} {_num(m['i5'], '+9.4f')} "
                  f"{_num(m['ratio'], '6.2f', 6)} {_num(m['dominance'], '6.2f', 6)} "
                  f"{_num(g.get('i3'), '+9.4f')} {_num(g.get('i5'), '+9.4f')}")

    failed = [k for k, c in numbers["checks"].items() if not c["pass"]]
    print("\n" + ("ALL CHECKS PASS" if not failed else f"FAILED: {failed}"))
    return 0 if not failed else 1
