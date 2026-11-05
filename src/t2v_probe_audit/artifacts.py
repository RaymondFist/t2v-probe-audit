"""Figures and table source data, rendered from the analysis output.

figures/fig1_sensitivity_specificity.png : order sensitivity against 1 - R
figures/fig2_density.png                 : support and ratio against density
tables/table1_probe_suite.csv / .md      : the six interventions
tables/table2_metric_response.csv / .md  : the ten metrics in two regimes

The tables are generated: table 1 from probes.PROBE_SPECS, table 2 from
output/analysis/numbers.json. The CSV files are the machine-readable source.
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import FIGURES_DIR, NUMBERS_JSON, TABLES_DIR
from .probes import PROBE_SPECS

REGIMES = [("floor33", "floor33, 16 frames @ 8 fps"),
           ("gen2", "gen2, 48 frames @ 24 fps")]
DENSITY_KEYS = ["floor33", "gen2_k3", "gen2"]
DENSITY_LABELS = ["floor33\n16f @ 8fps", "gen2 /3\n16f @ 8fps", "gen2\n48f @ 24fps"]
MARKERS = {
    "mafd": "o", "ssim_instability": "s", "warp_error": "^", "mcs_disc": "D",
    "mcs_warp": "v", "mcs_mag": "P", "mcs": "X", "dino_pair_mean": "*",
    "dino_pair_min": "<", "dino_pair_std": ">",
}

# Metric keys and labels for table 2.
TABLE2_ROWS = [
    ("mafd", "MAFD"), ("ssim_instability", "SSIM instability"),
    ("warp_error", "Warp error"), ("mcs_disc", "MCS direction"),
    ("mcs_warp", "MCS warp"), ("mcs_mag", "MCS magnitude"),
    ("mcs", "MCS composite"), ("dino_pair_mean", "DINOv2 mean"),
    ("dino_pair_min", "DINOv2 min"), ("dino_pair_std", "DINOv2 std"),
]
# Guarded columns exist only for the four flow channels.
GUARDED_OF = {
    "mcs_disc": "mcs_disc_guard", "mcs_warp": "mcs_warp_guard",
    "mcs_mag": "mcs_mag_guard", "mcs": "mcs_guard",
}


def load_numbers(path: Path = NUMBERS_JSON):
    if not path.exists():
        raise SystemExit(f"missing {path}; run the analysis first")
    return json.loads(path.read_text(encoding="utf-8"))


def _delta(value):
    return f"{value:.4f}" if value is not None else "n/a"


def _ratio(value):
    return f"{value:.2f}" if value is not None else "n/a"


# --------------------------------------------------------------------------- figures

def figure_sensitivity(numbers):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.35), sharey=True)
    for ax, (gen, title) in zip(axes, REGIMES):
        block = numbers["per_regime"][gen]
        for key, m in block["metrics"].items():
            if m["i3"] is None or m["ratio"] is None:
                continue
            ax.scatter(abs(m["i3"]), max(0.0, 1.0 - m["ratio"]),
                       marker=MARKERS.get(key, "o"), s=26, label=m["label"])
        ax.axhline(0.5, color="0.6", ls="--", lw=0.8)
        ax.set_xlabel(r"order sensitivity  $|\Delta_{\mathrm{bad}}(I3)|$", fontsize=8.5)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7.5)
        ax.set_ylim(-0.04, 1.04)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel(r"$1-R$  (R = $|\Delta(I5)|/|\Delta(I3)|$)", fontsize=8.5)
    axes[0].annotate("floor = $R\\geq1$", xy=(0.03, 0.03), fontsize=7, color="0.35")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.legend(handles, labels, ncol=5, fontsize=7, loc="lower center",
               frameon=False, bbox_to_anchor=(0.5, 0.01))
    out = FIGURES_DIR / "fig1_sensitivity_specificity.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print("wrote", out)


def figure_density(numbers):
    keys = [k for k in DENSITY_KEYS if k in numbers["per_regime"]]
    labels = [DENSITY_LABELS[DENSITY_KEYS.index(k)] for k in keys]
    cov = [numbers["per_regime"][k]["coverage"]["disc_coverage_mean"] for k in keys]
    ratio = [numbers["per_regime"][k]["metrics"]["warp_error"]["ratio"] for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.15))
    axes[0].bar(range(len(keys)), cov, color="#4472a8", width=0.55)
    axes[0].set_xticks(range(len(keys)))
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("direction-channel coverage", fontsize=8.5)
    axes[0].tick_params(axis="y", labelsize=7.5)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("(a) support on the unmodified clip", fontsize=9)

    axes[1].plot(range(len(keys)), ratio, "o-", color="#a84b3f")
    axes[1].axhline(1.0, color="0.5", ls="--", lw=0.8)
    axes[1].set_xticks(range(len(keys)))
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("warp-error ratio R", fontsize=8.5)
    axes[1].tick_params(axis="y", labelsize=7.5)
    axes[1].set_title("(b) specificity, parity marked", fontsize=9)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    out = FIGURES_DIR / "fig2_density.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print("wrote", out)


# ----------------------------------------------------------------------------- tables

def table_probe_suite():
    """Table 1: the six interventions and their exact pixel-level effect."""
    columns = ["probe", "operation", "pixel_values", "frame_order",
               "adjacent_pairs", "role", "reported"]
    rows = [[f"{p['id']} {p['name']}", p["operation"], p["pixel_values"],
             p["frame_order"], p["adjacent_pairs"], p["role"],
             "yes" if p["reported"] else "no"] for p in PROBE_SPECS]
    csv_path = TABLES_DIR / "table1_probe_suite.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        w.writerows(rows)
    md = ["| " + " | ".join(columns) + " |",
          "|" + "|".join(["---"] * len(columns)) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    (TABLES_DIR / "table1_probe_suite.md").write_text("\n".join(md) + "\n",
                                                      encoding="utf-8")
    print("wrote", csv_path)


def table_metric_response(numbers):
    """Table 2: mean paired badness change and R for the ten metrics."""
    columns = ["metric", "I3_8fps", "I5_8fps", "R_8fps",
               "I3_24fps", "I5_24fps", "R_24fps",
               "guarded_I3_8fps", "guarded_I5_8fps"]
    rows = []
    for key, label in TABLE2_ROWS:
        a = numbers["per_regime"]["floor33"]["metrics"][key]
        b = numbers["per_regime"]["gen2"]["metrics"][key]
        g = numbers["per_regime"]["floor33"]["guarded"].get(GUARDED_OF.get(key, ""))
        rows.append([
            label, _delta(a["i3"]), _delta(a["i5"]), _ratio(a["ratio"]),
            _delta(b["i3"]), _delta(b["i5"]), _ratio(b["ratio"]),
            _delta(g["i3"]) if g else "N/A",
            _delta(g["i5"]) if g else "N/A",
        ])
    csv_path = TABLES_DIR / "table2_metric_response.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        w.writerows(rows)
    header = ["Metric", "I3, 8 fps", "I5, 8 fps", "R", "I3, 24 fps",
              "I5, 24 fps", "R", "guarded I3", "guarded I5"]
    md = ["| " + " | ".join(header) + " |",
          "|" + "|".join(["---"] * len(header)) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    (TABLES_DIR / "table2_metric_response.md").write_text("\n".join(md) + "\n",
                                                         encoding="utf-8")
    print("wrote", csv_path)


def build_all(numbers=None):
    numbers = numbers or load_numbers()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_sensitivity(numbers)
    figure_density(numbers)
    table_probe_suite()
    table_metric_response(numbers)
