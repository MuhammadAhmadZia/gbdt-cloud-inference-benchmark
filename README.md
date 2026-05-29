# Precision and Latency Benchmarking of GBDT Inference Across Cloud CPU Environments for Sustainable Urban Services

Companion repository to the paper. Contains the trained models' generation
script, the main benchmark harness, the four Phase 3 follow-up experiments,
and the aggregation / analysis pipeline needed to reproduce every figure and
table.

DOI: `10.5281/zenodo.20311801`

---

## Overview

The study runs in three phases (Fig. 1 of the paper):

* **Phase 1** &mdash; one-time training of three GBDT engines (LightGBM, XGBoost,
  CatBoost) on two urban tasks (Olist delivery classification, NYC Yellow
  Taxi trip-duration regression).
* **Phase 2** &mdash; inference benchmarking of the trained models across four
  free-tier cloud CPU environments and five precision profiles
  (FP64, FP32, FP16, BF16, INT8). 6,000 timed runs (50 reps each).
* **Phase 3** &mdash; four follow-up experiments on a single platform (Colab,
  unless noted) that close the gaps identified during review: in-range 8-bit
  quantization (accuracy recovery), single-thread latency control, real
  compiled inference runtimes (Treelite + ONNX Runtime), and the
  variance-decomposition / effect-size analysis.

---

## Contents

```
code/
  training_colab.py                 Phase 1: train and save the 6 GBDT models
  inference_harness.py              Phase 2: main precision benchmark (per platform)
  aggregate_results.py              combine the 4 platform JSONs + Welch's t-tests
  exp1_proper_quantization.py       Phase 3: in-range 8-bit recovery (Sec. 4.4)
  exp2_single_thread.py             Phase 3: single-thread latency control (Sec. 4.3)
  exp4_real_runtime_benchmark.py    Phase 3: Treelite + ONNX Runtime (Sec. 4.5)
  anova_variance_decomposition.py   four-way ANOVA on log-latency (Sec. 4.3)
  cohens_d.py                       Cohen's d for FP64 vs each precision

results/
  results_broadwell_colab.json      Phase 2 raw results per platform
  results_skylake_kaggleb.json
  results_epyc_codespaces.json
  results_huggingface_cpu.json
  combined_results.csv              Phase 2 aggregated table
  pairwise_significance.txt         Welch's t-tests, Holm-Bonferroni corrected

  exp1_results_colab.json           Phase 3 outputs
  exp2_single_thread_colab.json
  exp2_single_thread_kaggle.json
  exp2_single_thread_codespaces.json
  exp2_single_thread_huggingface_cpu.json
  exp4_results_runtime_colab.json

  anova_decomposition.txt           eta-squared by factor
  effect_sizes_cohens_d.txt         Cohen's d table per (platform, engine, dataset)
```

---

## Reproducing

### Environment

Python 3.12. Install:

```bash
pip install numpy pandas scipy scikit-learn statsmodels \
            lightgbm==4.6.0 xgboost==3.2.0 catboost==1.2.10 \
            codecarbon pyarrow \
            treelite tl2cgen onnxruntime onnxmltools skl2onnx
```

The last line is for the Phase 3 compiled-runtime experiment; skip it if you
only run Phases 1 and 2.

### Phase 1 &mdash; Training (one-time, on Colab)

```bash
python code/training_colab.py
```

Trains all six (engine x dataset) GBDT models and writes them, along with
the held-out test data, to your Google Drive at
`/content/drive/MyDrive/paper2_benchmark/`.

### Phase 2 &mdash; Main precision benchmark (run on each platform)

```bash
python code/inference_harness.py
```

Per platform (Colab, Kaggle, Codespaces, Hugging Face Spaces), produces one
`results_<platform>.json` with 1,500 timed runs. After running on all four,
aggregate:

```bash
python code/aggregate_results.py
```

This writes `combined_results.csv` and `pairwise_significance.txt`.

### Phase 3 &mdash; Follow-up experiments

```bash
# in-range 8-bit quantization (accuracy recovery)
python code/exp1_proper_quantization.py

# single-thread control (run on each platform; edit PLATFORM_TAG/BASE_DIR)
python code/exp2_single_thread.py

# Treelite + ONNX Runtime comparison
python code/exp4_real_runtime_benchmark.py
```

Each script writes its own JSON to `results/`, in the same schema as the
Phase 2 outputs.

### Analysis (variance decomposition and effect sizes)

```bash
python code/anova_variance_decomposition.py    # -> results/anova_decomposition.txt
python code/cohens_d.py                        # -> results/effect_sizes_cohens_d.txt
```

Both read the four Phase-2 JSONs and reproduce the numbers cited in
Sections 4.1 and 4.3.

---

## Supplementary materials

The following files contain the supporting data behind the paper's claims:

* `combined_results.csv` &mdash; one row per timed configuration with mean
  latency, 95% CI, energy estimate, and accuracy.
* `pairwise_significance.txt` &mdash; Welch's t-test results for FP64 against
  every other precision, Holm-Bonferroni corrected across the 96 comparisons.
* `effect_sizes_cohens_d.txt` &mdash; Cohen's d for the same comparisons,
  with magnitude labels (negligible / small / moderate / large).
* `anova_decomposition.txt` &mdash; the four-way ANOVA eta-squared
  attribution by engine, dataset, platform, and precision; reproduces the
  79% / 8% / 5% / <1% figures in Section 4.3.

---

## Notes for reviewers and re-runners

* Free-tier cloud environments are shared and noisy. Per-call timings include
  tenant interference; we mitigate with 50-run averaging and 95% confidence
  intervals, which is the strongest control available without privileged
  access to the host.
* Hardware performance counters and direct package-energy registers are not
  exposed to user-space code on these platforms, so energy is estimated via
  CodeCarbon (RAPL passthrough where the platform permits it). This matches
  the deployment reality for cost-sensitive operators using shared cloud
  infrastructure; cf. Henderson et al., JMLR 2020.
* All accuracy values are deterministic given the saved test data, so they
  reproduce exactly across platforms. Latency values are platform-dependent
  by construction.
