# t2v-probe-audit

A controlled-probe audit of temporal-consistency metrics for text-to-video generation. Six interventions with exactly known pixel-level effects test whether ten metrics (pixel, flow, embedding) measure what their names claim. Includes a support-aware repair, a density control, and scripts that regenerate every reported number and figure.

## Contents

This repository measures construct and discriminant validity of
temporal-consistency metrics on the public ECTV text-to-video clips. Six
interventions are defined on decoded frames. The reported subset is I1, I3 and
I5: I1 reverses the clip, I3 permutes the frame order, and I5 duplicates every
frame. I3 and I5 both preserve the multiset of pixel values; I3 changes the
order and I5 does not.

## Layout

```
t2v-probe-audit/
├── data/                 inputs only; the analysis never writes here
│   ├── raw/              ECTV corpus, one folder per generator (not redistributed)
│   └── manifest.json     what was read: clips, fps, frames, resolution, sizes
├── output/               everything the experiments produce
│   ├── metrics/          per-clip records: regime.json, regime_control.json, multigen.json
│   ├── analysis/         numbers.json (reported statistics), checks.json (invariants)
│   └── logs/             one log per run
├── figures/              rendered figures (fig1, fig2)
├── tables/               table source data: CSV (machine) and Markdown (rendered)
├── scripts/              one-click entry points and manuscript token substitution
├── src/t2v_probe_audit/  the package
└── tests/                unit tests for the primitives
```

Source modules, in dependency order:

| Module | Responsibility |
|---|---|
| `config.py` | paths and fixed constants (flow parameters, guard thresholds, metric polarity) |
| `corpus.py` | listing, deterministic sampling, decoding, 2.0 s truncation, header reads |
| `probes.py` | the six interventions and `PROBE_SPECS`, the metadata used for Table 1 |
| `metrics/pixel.py` | working-resolution conversion and local SSIM |
| `metrics/flow.py` | dense flow warping and the partial-affine global-motion fit |
| `metrics/guard.py` | the degenerate-fit guard |
| `metrics/sequence.py` | per-sequence assembly of the raw and guarded channels |
| `embedding.py` | DINOv2 ViT-S/14 embeddings and pair statistics |
| `stats.py` | badness deltas, specificity ratio, dominance, bootstrap, Kendall tau |
| `pipeline.py` | one clip through a subset of the probe suite |
| `runner.py` | the three experiments, with process-level parallelism |
| `analysis.py` | recomputes `numbers.json` and `checks.json` from the per-clip records |
| `artifacts.py` | figures and table source data |
| `manifest.py` | the corpus provenance record |
| `cli.py` | the command line entry point |

## Install

```
python -m pip install -r requirements.txt
```

DINOv2 ViT-S/14 is loaded from the local torch-hub clone at
`~/.cache/torch/hub/facebookresearch_dinov2_main`; no network access is needed
at run time.

## Data

The corpus is the public **ECTV** video release of the EvalCrafter benchmark
(Hugging Face `RaphaelLiu/EvalCrafter_T2V_Dataset`). It is third-party material
and is not redistributed here. Point `data/raw` at a local copy, without copying
any file:

```
powershell -ExecutionPolicy Bypass -File scripts/link_data.ps1 -Source "D:\path\to\ectv_videos"
```

`data/raw` must contain one directory per generator (`floor33`, `gen2`, `pika`,
`modelscope`, `zeroscope`) with `.mp4` clips. Setting `TPA_DATA_ROOT` overrides
the location. Only the released clips are decoded; no clip is generated, edited
or synthesised.

## One-click runs

Run the experiments and the analysis:

```
powershell -ExecutionPolicy Bypass -File scripts/run_experiments.ps1
bash scripts/run_experiments.sh
```

This builds `data/manifest.json` if missing, runs the three experiments, and
writes `output/metrics/*.json` plus `output/analysis/numbers.json` and
`checks.json`. Options: `-N 10 -Workers 4` for a smoke run.

Render the figures and the table source data:

```
powershell -ExecutionPolicy Bypass -File scripts/make_artifacts.ps1
bash scripts/make_artifacts.sh
```

This reads `output/analysis/numbers.json` and writes `figures/*.png` and
`tables/table1_probe_suite.*`, `tables/table2_metric_response.*`.

Refresh the manuscript tokens from the result files:

```
python scripts/fill_manuscript.py
```

The script refuses to finish while any `{{token}}` is unresolved.

The same steps are available through the package entry point:

```
python -m t2v_probe_audit all          # manifest + run + analyze + artifacts
python -m t2v_probe_audit run          # experiments only
python -m t2v_probe_audit analyze      # numbers and checks only
python -m t2v_probe_audit artifacts    # figures and tables only
```

Set `PYTHONPATH=src` (the one-click scripts do this) or install with
`pip install -e .`.

## Tests

```
python -m pytest tests -q
```

## Output contract

* `output/metrics/regime.json`: 60 clips for each of floor33 (16 frames at 8 fps)
  and gen2 (48 frames at 24 fps), each with base metrics and its I1, I3 and I5 variants.
* `output/metrics/regime_control.json`: gen2 kept every third frame: the floor33
  density on one generator.
* `output/metrics/multigen.json`: the five-generator consequence analysis, with
  the pre-repair and repaired composite, coverage and the ranking change.
* `output/analysis/numbers.json`: the statistics reported in the manuscript,
  derived from the records above.
* `output/analysis/checks.json`: pass/fail results for the stated invariants.
* `figures/`, `tables/`: generated deliverables.

## Method summary

* Interventions act on decoded frames in memory; no clip is re-encoded.
* I5 (0.5x playback by duplication) changes no pixel value and does not reorder
  frames.
* The guarded channel does not fit a global affine model to a near-static
  frame pair (median displacement < 0.1 px, or induced displacement >
  1.2 × observed + 0.5 px) and bridges the direction chain past rejected pairs.
* Undefined channels are reported as `null` and coverage is reported beside the
  score.