"""
exp2_single_thread.py

Phase 3 follow-up experiment (Section 4.3 of the paper).

Question
--------
Phase 2 measures inference latency under each engine's default thread setting
(all available cores). This conflates CPU architecture with the number of
cores the platform happens to expose. To separate them, we re-run FP32
inference for each (dataset, engine) with the thread count forced to 1.

If most of the cross-platform spread is core count, single-thread results
should converge across platforms with similar per-core architecture. The
paper finds the LightGBM-Olist cross-platform ratio falls from ~2.9x at
default threading to ~1.6x at one thread, and Colab and Kaggle (nominally
identical Xeons, different core counts) converge to within a few percent.

How to reproduce
----------------
1. Trained models and held-out test data are expected at the same paths as
   the main benchmark.
2. Run this script on each target platform, changing PLATFORM_TAG and
   BASE_DIR per platform (same convention as inference_harness.py):

     PLATFORM_TAG = "colab"           BASE_DIR = "/content/drive/MyDrive/paper2_benchmark"
     PLATFORM_TAG = "kaggle"          BASE_DIR = "/kaggle/input/.../paper2_benchmark"
     PLATFORM_TAG = "codespaces"      BASE_DIR = "/workspaces/.../paper2_benchmark"
     PLATFORM_TAG = "huggingface"     BASE_DIR = "/home/user/paper2_benchmark"

3. Output: exp2_single_thread_<platform>.json
   Two entries per (dataset, engine): one at default threading, one at 1 thread.

Threading is set per call:
   LightGBM:  predict(..., num_threads=1)        OR  num_threads=-1 for default
   XGBoost:   booster.set_param("nthread", 1)    OR  unset
   CatBoost:  set_thread_count(1)                OR  set_thread_count(-1)

Notes for end users
-------------------
- 3 warm-up + 50 timed calls on a 10,000-row batch (matches Phase 2).
- This is FP32 only. Adding more precisions is straightforward (mirror the
  inner loop of inference_harness.py), but the paper's claim only needs FP32.
- Set OMP_NUM_THREADS=1 in the environment for an extra safety net on
  platforms that ignore per-call thread settings.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor


# ---------- config (edit per platform) ----------
PLATFORM_TAG = "colab"
BASE_DIR     = Path("/content/drive/MyDrive/paper2_benchmark")
RESULTS_FILE = Path(f"/content/exp2_single_thread_{PLATFORM_TAG}.json")
BATCH_SIZE   = 10_000
WARMUP       = 3
TIMED        = 50


# ---------- helpers ----------
def task_of(dataset):
    return "classification" if dataset == "olist" else "regression"


def load_test_data(dataset):
    Xdf = pd.read_parquet(BASE_DIR / "test_data" / f"{dataset}_X_test.parquet").iloc[:BATCH_SIZE]
    y   = pd.read_parquet(BASE_DIR / "test_data" / f"{dataset}_y_test.parquet").values.ravel()[:BATCH_SIZE]
    X   = Xdf.values.astype(np.float32)
    return Xdf, X, y


def load_model(dataset, engine):
    mdir = BASE_DIR / "saved_models" / dataset
    if engine == "lightgbm":
        return lgb.Booster(model_file=str(mdir / "lgbm.txt"))
    if engine == "xgboost":
        b = xgb.Booster()
        b.load_model(str(mdir / "xgb.json"))
        return b
    is_clf = task_of(dataset) == "classification"
    m = CatBoostClassifier() if is_clf else CatBoostRegressor()
    m.load_model(str(mdir / "catboost.cbm"))
    return m


def predict_fn(engine, model, X_df, X, threads):
    """Return a no-arg function that runs one prediction with the given thread count.
       threads = -1 means each engine's default ('all cores')."""
    if engine == "lightgbm":
        nt = threads if threads != -1 else 0      # lightgbm 0 = all cores
        return lambda: model.predict(X, num_threads=nt)
    if engine == "xgboost":
        if threads != -1:
            model.set_param({"nthread": threads})
        else:
            # restore default (0 means all)
            model.set_param({"nthread": 0})
        dm = xgb.DMatrix(pd.DataFrame(X, columns=X_df.columns))
        return lambda: model.predict(dm)
    # CatBoost
    nt = threads if threads != -1 else -1         # catboost -1 = all cores
    model.set_param("thread_count", nt)
    if isinstance(model, CatBoostClassifier):
        return lambda: model.predict_proba(X)[:, 1]
    return lambda: model.predict(X)


def time_calls(fn):
    for _ in range(WARMUP):
        fn()
    samples = np.empty(TIMED, dtype=np.float64)
    for i in range(TIMED):
        t0 = time.perf_counter()
        fn()
        samples[i] = (time.perf_counter() - t0) * 1000.0
    mean = float(samples.mean())
    std  = float(samples.std(ddof=1))
    half = float(st.t.ppf(0.975, df=TIMED - 1) * std / np.sqrt(TIMED))
    return {"n_reps": TIMED, "mean_ms": mean, "std_ms": std,
            "ci95_half_ms": half, "samples_ms": samples.tolist()}


def load_results():
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {"platform": PLATFORM_TAG, "experiments": []}


def save_results(d):
    RESULTS_FILE.write_text(json.dumps(d, indent=2))


def already_done(d, ds, eng, label):
    return any(e["dataset"] == ds and e["model"] == eng and e["precision"] == label
               for e in d["experiments"])


# ---------- main ----------
def run_one(dataset, engine, threads, label, results):
    if already_done(results, dataset, engine, label):
        print(f"  skip {dataset}/{engine}/{label}")
        return
    print(f"  -> {dataset}/{engine}/{label}")
    X_df, X, _ = load_test_data(dataset)
    model      = load_model(dataset, engine)
    fn         = predict_fn(engine, model, X_df, X, threads)
    timing     = time_calls(fn)
    results["experiments"].append({
        "dataset":   dataset,
        "model":     engine,
        "precision": label,
        "n_reps":    timing["n_reps"],
        "latency_ms": {
            "mean":      timing["mean_ms"],
            "std":       timing["std_ms"],
            "ci95_half": timing["ci95_half_ms"],
            "samples":   timing["samples_ms"],
        },
    })
    save_results(results)
    print(f"     latency {timing['mean_ms']:7.2f} +/- {timing['ci95_half_ms']:.2f} ms")


def main():
    try:
        from google.colab import drive  # type: ignore
        drive.mount('/content/drive', force_remount=False)
    except Exception:
        pass

    results = load_results()
    for ds in ("olist", "nyc_taxi"):
        for eng in ("lightgbm", "xgboost", "catboost"):
            run_one(ds, eng, threads=-1, label="fp32_default", results=results)
            run_one(ds, eng, threads=1,  label="fp32_1thread", results=results)

    print(f"\nDone. Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
