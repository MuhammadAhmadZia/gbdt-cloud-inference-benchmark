"""
anova_variance_decomposition.py

Phase 2 / aggregation analysis (Section 4.3 of the paper).

Computes a four-way ANOVA on log-latency across all main-benchmark runs and
prints the eta-squared decomposition by factor (engine, dataset, platform,
precision). Also reports the FP-only decomposition (excluding INT8), which
isolates the cross-platform spread that does NOT depend on the dramatic
INT8 latency cut.

The paper cites:
    engine    79.3%
    dataset    8.3%
    platform   4.8%
    precision  0.3%
    residual   7.3%

    FP-only:  platform 4.5%   precision 0.04%   (ratio ~100x)

How to reproduce
----------------
1. Place the four per-platform result files from Phase 2 in results/:
       results/results_broadwell_colab.json
       results/results_skylake_kaggleb.json
       results/results_epyc_codespaces.json
       results/results_huggingface_cpu.json

2. Run:
       python code/anova_variance_decomposition.py

3. Output: results/anova_decomposition.txt

Notes for end users
-------------------
- Latency is log-transformed because the four engines/datasets span two
  orders of magnitude; an additive ANOVA on raw latency would be dominated
  by mean shifts rather than relative variance attribution.
- We report eta-squared (SS_factor / SS_total) which sums to 1 across
  factors + residual.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


RESULTS_DIR = Path("results")
OUT_FILE    = Path("results/anova_decomposition.txt")

# Map platform -> result file (edit if your file names differ)
PLATFORM_FILES = {
    "colab":       RESULTS_DIR / "results_broadwell_colab.json",
    "kaggle":      RESULTS_DIR / "results_skylake_kaggleb.json",
    "codespaces":  RESULTS_DIR / "results_epyc_codespaces.json",
    "huggingface": RESULTS_DIR / "results_huggingface_cpu.json",
}


def load_long_dataframe():
    """One row per (platform, engine, dataset, precision, rep). Latency in ms."""
    rows = []
    for platform, path in PLATFORM_FILES.items():
        if not path.exists():
            print(f"warning: {path} not found, skipping {platform}")
            continue
        data = json.loads(path.read_text())
        for exp in data["experiments"]:
            samples = exp.get("latency_ms", {}).get("samples")
            if not samples:
                # Fallback: use mean only as a single observation. Less powerful.
                samples = [exp["latency_ms"]["mean"]]
            for s in samples:
                rows.append({
                    "platform":  platform,
                    "engine":    exp["model"],
                    "dataset":   exp["dataset"],
                    "precision": exp["precision"],
                    "latency":   float(s),
                })
    df = pd.DataFrame(rows)
    df["log_latency"] = np.log(df["latency"])
    return df


def anova_table(df, label, out_lines):
    formula = "log_latency ~ C(engine) + C(dataset) + C(platform) + C(precision)"
    model   = smf.ols(formula, data=df).fit()
    aov     = sm.stats.anova_lm(model, typ=2)
    ss_total = aov["sum_sq"].sum()
    aov["eta_sq"]    = aov["sum_sq"] / ss_total
    aov["pct"]       = aov["eta_sq"] * 100
    out_lines.append(f"\n=== {label} ===\n")
    out_lines.append(f"N = {len(df)} observations\n")
    out_lines.append(aov[["sum_sq", "df", "F", "PR(>F)", "eta_sq", "pct"]].to_string())
    out_lines.append("\n")


def main():
    df = load_long_dataframe()
    if df.empty:
        raise SystemExit("No data loaded - check PLATFORM_FILES paths.")

    out_lines = ["Four-way ANOVA on log-latency.\n"]

    # All five precisions
    anova_table(df, "All precisions (FP64, FP32, FP16, BF16, INT8)", out_lines)

    # FP profiles only
    fp_only = df[df["precision"].isin(["fp64", "fp32", "fp16", "bf16"])]
    if not fp_only.empty:
        anova_table(fp_only, "Floating-point profiles only (excluding INT8)", out_lines)

    text = "".join(out_lines)
    print(text)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(text)
    print(f"\nWrote {OUT_FILE}")


if __name__ == "__main__":
    main()
