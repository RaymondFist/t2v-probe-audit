"""Substitute the numeric tokens in Manuscript.md from the result files.

Tokens have the form {{kind|args...}} and are resolved from
output/analysis/numbers.json (regime audit) and output/metrics/multigen.json
(consequence analysis). Execution fails if any token remains unresolved.
"""

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGIME_KEY = {"f33": "floor33", "g2": "gen2", "k3": "gen2_k3"}


def load(path: Path):
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_delta(value):
    return f"{value:.4f}" if value is not None else "n/a"


def fmt_ratio(value):
    return f"{value:.2f}" if value is not None else "n/a"


def fmt_pct(value):
    return f"{value:.0%}" if value is not None else "n/a"


def fmt_plain(value, nd=3):
    return f"{value:.{nd}f}" if value is not None else "n/a"


def fmt_max(value):
    if value is None:
        return "n/a"
    if value == 0:
        return "0"
    return f"{value:.1e}"


def make_resolver(numbers, multi):
    regs = numbers["per_regime"]

    def resolve(raw):
        parts = raw.split("|")
        head = parts[0]

        if head in ("f33", "g2", "k3"):
            regime, metric, kind = parts
            block = regs[REGIME_KEY[regime]]["metrics"][metric]
            if kind == "I3":
                return fmt_delta(block["i3"])
            if kind == "I5":
                return fmt_delta(block["i5"])
            if kind == "R":
                return fmt_ratio(block["ratio"])
            if kind == "D":
                return fmt_pct(block["dominance"])
            raise KeyError(raw)

        if head in ("g", "gw", "gm", "gc"):
            _, regime, kind = parts
            key = {"g": "mcs_disc_guard", "gw": "mcs_warp_guard",
                   "gm": "mcs_mag_guard", "gc": "mcs_guard"}[head]
            block = regs[REGIME_KEY[regime]]["guarded"][key]
            if kind == "I3":
                return fmt_delta(block["i3"])
            if kind == "I5":
                return fmt_delta(block["i5"])
            if kind == "lo":
                return fmt_delta(block["i3_ci95"][0])
            if kind == "hi":
                return fmt_delta(block["i3_ci95"][1])
            if kind == "n":
                return str(block["n3"])
            raise KeyError(raw)

        if head in ("def", "defc", "undef", "cov"):
            _, regime = parts
            block = regs[REGIME_KEY[regime]]
            cov = block["coverage"]
            if head == "def":
                return str(cov["disc_defined"])
            if head == "defc":
                return str(cov["composite_defined"])
            if head == "undef":
                return str(block["n"] - cov["disc_defined"])
            if head == "cov":
                return fmt_plain(cov["disc_coverage_mean"])
            raise KeyError(raw)

        if head == "ap":
            kind = parts[1]
            block = regs["floor33"]
            if kind == "max":
                return fmt_max(block["allpairs_max_abs_delta"])
            if kind == "n":
                return str(block["allpairs_n"])
            if kind == "count":
                return str(block["allpairs_invariant_count"])
            raise KeyError(raw)

        if head == "i1":
            kind = parts[1]
            block = regs["floor33"]
            if kind == "max":
                return fmt_max(block["i1_max_abs_delta"])
            if kind == "n":
                return str(block["i1_n"])
            raise KeyError(raw)

        if head == "guard":
            kind = parts[1]
            block = regs["floor33"]
            if kind == "n":
                return str(block["guard_i5_n"])
            if kind == "max":
                return fmt_max(block["guard_i5_max_abs_delta"])
            raise KeyError(raw)

        if head == "mg":
            if len(parts) == 2:
                kind = parts[1]
                con = multi["consequence"]
                if kind == "tau":
                    return fmt_plain(con["kendall_tau"], 2)
                if kind == "gap":
                    return fmt_plain(con["between_generator_gap"], 4)
                if kind == "maxart":
                    return fmt_plain(con["max_abs_artifact"], 4)
                if kind == "ratio":
                    return fmt_plain(con["artifact_over_gap"], 2)
                if kind == "swapgap":
                    pre = con["ranking_pre"]
                    return fmt_plain(abs(pre["modelscope"] - pre["zeroscope"]), 4)
                if kind == "inv":
                    return str(sum(multi["generators"][g]["inverted_pairs"]
                                   for g in multi["generators"]))
                raise KeyError(raw)
            _, gen, kind = parts
            block = multi["generators"][gen]
            if kind == "delta":
                return fmt_delta(block["reported_post"] - block["reported_pre"])
            if kind == "covpre":
                return fmt_plain(block["coverage_pre"], 2)
            if kind == "covpost":
                return fmt_plain(block["coverage_post"], 2)
            if kind == "invpct":
                return fmt_pct(block["pairwise_inversion"])
            raise KeyError(raw)

        raise KeyError(raw)

    return resolve


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", default=str(PROJECT_ROOT.parent / "Manuscript.md"))
    ap.add_argument("--out", default=str(PROJECT_ROOT.parent / "Manuscript_filled.md"))
    ap.add_argument("--numbers", default=str(PROJECT_ROOT / "output" / "analysis"
                                             / "numbers.json"))
    ap.add_argument("--multigen", default=str(PROJECT_ROOT / "output" / "metrics"
                                              / "multigen.json"))
    args = ap.parse_args(argv)

    numbers = load(Path(args.numbers))
    multi = load(Path(args.multigen))
    resolve = make_resolver(numbers, multi)
    text = Path(args.template).read_text(encoding="utf-8")

    unresolved = []

    def sub(match):
        try:
            return resolve(match.group(1))
        except Exception:
            unresolved.append(match.group(0))
            return match.group(0)

    total = len(re.findall(r"\{\{(.+?)\}\}", text))
    out_text = re.sub(r"\{\{(.+?)\}\}", sub, text)
    out = Path(args.out)
    out.write_text(out_text, encoding="utf-8")

    if unresolved:
        print("UNRESOLVED TOKENS:", sorted(set(unresolved)))
        return 1
    print(f"filled {total} tokens -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
