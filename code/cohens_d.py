"""
cohens_d.py

Phase 2 / aggregation analysis (referenced in Section 3.3 and 4.1 of the paper).

Computes Cohen's d for each (platform, engine, dataset) latency comparison
between FP64 (baseline) and every other precision profile (FP32, FP16, BF16,
INT8). Combined with the Welch's t-test p-values produced by aggregate_results.py
this gives reviewers the full picture: which statistically significant
differences are also practically large.

Standard interpretation:
    |d| < 0.2   negligible
    |d| ~ 0.5   moderate
    |d| > 0.8   large

The paper reports that FP-profile comparisons consistently yield small d
(< 0.5), while INT8 vs FP64 yields very large d (> 5 for LightGBM on Olist).

How to reproduce
----------------
1. Place the four per-platform result files in results/ (same as
   anova_variance_decomposition.py).

2. Run:
       python code/cohens_d.py

3. Output: results/effect_sizes_cohens_d.txt
   One row per (platform, engine, dataset, comparison).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
OUT_FILE    = Path("results/effect_sizes_cohens_d.txt")

PLATFORM_FILES = {
    "colab":       RESULTS_DIR / "results_broadwell_colab.json",
    "kaggle":      RESULTS_DIR / "results_skylake_kaggleb.json",
    "codespaces":  RESULTS_DIR / "results_epyc_codespaces.json",
    "huggingface": RESULTS_DIR / "results_huggingface_cpu.json",
}


def cohens_d(a, b):
    """Pooled-sd Cohen's d between two samples a and b."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    s2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    s  = np.sqrt(s2)
    if s == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / s)


def magnitude(d):
    ad = abs(d)
    if np.isnan(d):       return "n/a"
    if ad < 0.2:          return "negligible"
    if ad < 0.5:          return "small"
    if ad < 0.8:          return "moderate"
    return "large"


def load_samples(path):
    """Return {(engine, dataset, precision): samples_list}."""
    out = {}
    data = json.loads(Path(path).read_text())
    for exp in data["experiments"]:
        s = exp.get("latency_ms", {}).get("samples")
        if not s:
            continue
        out[(exp["model"], exp["dataset"], exp["precision"])] = s
    return out


def main():
    rows = []
    for platform, path in PLATFORM_FILES.items():
        if not path.exists():
            print(f"warning: {path} not found, skipping {platform}")
            continue
        samples = load_samples(path)
        engines  = sorted({k[0] for k in samples})
        datasets = sorted({k[1] for k in samples})
        for eng in engines:
            for ds in datasets:
                base = samples.get((eng, ds, "fp64"))
                if not base:
                    continue
                for prec in ("fp32", "fp16", "bf16", "int8"):
                    other = samples.get((eng, ds, prec))
                    if not other:
                        continue
                    d = cohens_d(base, other)
                    rows.append({
                        "platform":   platform,
                        "engine":     eng,
                        "dataset":    ds,
                        "comparison": f"fp64_vs_{prec}",
                        "cohens_d":   d,
                        "magnitude":  magnitude(d),
                    })

    if not rows:
        raise SystemExit("No samples loaded - check that result files contain "
                         "'latency_ms.samples' arrays (re-run with sample logging on).")

    df = pd.DataFrame(rows)
    text = df.to_string(index=False, float_format=lambda x: f"{x:+.3f}")
    print(text)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(text + "\n")
    print(f"\nWrote {OUT_FILE}")


if __name__ == "__main__":
    main()
