"""
exp1_proper_quantization.py

Phase 3 follow-up experiment (Section 4.4 of the paper).

Question
--------
The main benchmark (Phase 2) shows naive INT8 input casting collapses GBDT
accuracy. Is the collapse a property of 8-bit precision itself, or an artefact
of the way the cast is applied (quantized inputs compared against unquantized
thresholds)?

Three treatments per (dataset, engine)
--------------------------------------
fp64
    Baseline. Original FP64 inputs, original thresholds.

naive_int8
    The naive cast from Eq. (1) of the paper:
        x_q = round( ((x - x_min) / (x_max - x_min)) * 254 - 127 ) in [-127,127]
    Stored as int8, then re-expanded to float32 before prediction. Thresholds
    stay at their trained FP64 values, so the engine compares quantized inputs
    against unquantized thresholds. Accuracy collapses.

proper_dequant
    8-bit resolution kept in each feature's real value range. Inputs are
    snapped to 256 equally spaced levels between the feature's empirical
    min and max, but the values remain in their original scale - so the
    trained thresholds are still meaningful. Order-preserving. Accuracy
    returns close to the FP64 baseline. Latency does not improve, because
    the array is re-expanded to float32 before prediction.

How to reproduce
----------------
1. Trained models and held-out test data are expected at the same paths as
   the main benchmark (Phase 1 / inference_harness.py):

     BASE_DIR/
       saved_models/{olist,nyc_taxi}/{lgbm.txt, xgb.json, catboost.cbm}
       test_data/{olist,nyc_taxi}_X_test.parquet
       test_data/{olist,nyc_taxi}_y_test.parquet

2. On Google Colab, mount Drive and run:
        python code/exp1_proper_quantization.py

3. Output: /content/exp1_results_colab.json
   Same schema as results_*.json (one entry per (dataset, model, treatment)
   with mean latency, 95% CI, and accuracy).

Notes for end users
-------------------
- The script is idempotent: it skips configurations already present in the
  output JSON, so a partial run can be resumed.
- 3 warm-up calls + 50 timed calls per configuration on a 10,000-row batch
  (matching the main benchmark).
- Random seeds are not consumed: the quantization is deterministic given the
  saved test data.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.metrics import mean_squared_error, roc_auc_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor


# ---------- config ----------
BASE_DIR     = Path("/content/drive/MyDrive/paper2_benchmark")
RESULTS_FILE = Path("/content/exp1_results_colab.json")
BATCH_SIZE   = 10_000
WARMUP       = 3
TIMED        = 50
PLATFORM_TAG = "colab"


# ---------- I/O helpers ----------
def task_of(dataset):
    return "classification" if dataset == "olist" else "regression"


def load_test_data(dataset):
    Xdf = pd.read_parquet(BASE_DIR / "test_data" / f"{dataset}_X_test.parquet").iloc[:BATCH_SIZE]
    y   = pd.read_parquet(BASE_DIR / "test_data" / f"{dataset}_y_test.parquet").values.ravel()[:BATCH_SIZE]
    X   = Xdf.values.astype(np.float64)
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


# ---------- the three transforms ----------
def transform_fp64(X):
    return X.astype(np.float64)


def transform_naive_int8(X):
    """Equation (1) from the paper. Mismatched comparison with FP64 thresholds."""
    X = X.astype(np.float64)
    mn = X.min(axis=0, keepdims=True)
    mx = X.max(axis=0, keepdims=True)
    rng = np.where((mx - mn) == 0, 1.0, (mx - mn))
    q = np.round(((X - mn) / rng) * 254 - 127).clip(-127, 127).astype(np.int8)
    return q.astype(np.float32)


def transform_proper_dequant(X, levels=256):
    """In-range 8-bit: 256 evenly spaced levels per feature, kept in real units."""
    X = X.astype(np.float64)
    mn = X.min(axis=0, keepdims=True)
    mx = X.max(axis=0, keepdims=True)
    rng = np.where((mx - mn) == 0, 1.0, (mx - mn))
    step = rng / (levels - 1)
    q = np.round((X - mn) / step) * step + mn
    return q.astype(np.float32)


TRANSFORMS = {
    "fp64":           transform_fp64,
    "naive_int8":     transform_naive_int8,
    "proper_dequant": transform_proper_dequant,
}


# ---------- prediction ----------
def predict(engine, model, X_df, X):
    if engine == "lightgbm":
        return model.predict(X)
    if engine == "xgboost":
        # XGBoost needs feature names that match the trained model; pass the
        # DataFrame so column names propagate.
        dm = xgb.DMatrix(pd.DataFrame(X, columns=X_df.columns))
        return model.predict(dm)
    # CatBoost
    if isinstance(model, CatBoostClassifier):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def score(dataset, y_true, y_pred):
    y_pred = np.asarray(y_pred).ravel()
    if task_of(dataset) == "classification":
        return {"metric": "roc_auc", "value": float(roc_auc_score(y_true, y_pred))}
    return {"metric": "rmse", "value": float(np.sqrt(mean_squared_error(y_true, y_pred)))}


def time_calls(fn, n_warmup=WARMUP, n_timed=TIMED):
    for _ in range(n_warmup):
        fn()
    samples = np.empty(n_timed, dtype=np.float64)
    for i in range(n_timed):
        t0 = time.perf_counter()
        fn()
        samples[i] = (time.perf_counter() - t0) * 1000.0
    mean = float(samples.mean())
    std  = float(samples.std(ddof=1))
    half = float(st.t.ppf(0.975, df=n_timed - 1) * std / np.sqrt(n_timed))
    return {"n_reps": int(n_timed), "mean_ms": mean, "std_ms": std,
            "ci95_half_ms": half, "samples_ms": samples.tolist()}


# ---------- results file ----------
def load_results():
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {"platform": PLATFORM_TAG, "experiments": []}


def save_results(d):
    RESULTS_FILE.write_text(json.dumps(d, indent=2))


def already_done(d, dataset, engine, treatment):
    return any(e["dataset"] == dataset and e["model"] == engine and e["precision"] == treatment
               for e in d["experiments"])


# ---------- main ----------
def run_one(dataset, engine, treatment, results):
    if already_done(results, dataset, engine, treatment):
        print(f"  skip {dataset}/{engine}/{treatment}")
        return
    print(f"  -> {dataset}/{engine}/{treatment}")
    X_df, X, y  = load_test_data(dataset)
    model       = load_model(dataset, engine)
    Xq          = TRANSFORMS[treatment](X)
    fn          = lambda: predict(engine, model, X_df, Xq)

    y_pred = fn()
    acc    = score(dataset, y, y_pred)
    timing = time_calls(fn)

    results["experiments"].append({
        "dataset":   dataset,
        "model":     engine,
        "precision": treatment,
        "n_reps":    timing["n_reps"],
        "latency_ms": {
            "mean":      timing["mean_ms"],
            "std":       timing["std_ms"],
            "ci95_half": timing["ci95_half_ms"],
            "samples":   timing["samples_ms"],
        },
        "accuracy":  acc,
    })
    save_results(results)
    print(f"     latency {timing['mean_ms']:7.2f} ms   {acc['metric']}={acc['value']:.4f}")


def main():
    # Colab Drive mount (no-op off Colab)
    try:
        from google.colab import drive  # type: ignore
        drive.mount('/content/drive', force_remount=False)
    except Exception:
        pass

    results = load_results()
    for ds in ("olist", "nyc_taxi"):
        for eng in ("lightgbm", "xgboost", "catboost"):
            for treatment in ("fp64", "naive_int8", "proper_dequant"):
                try:
                    run_one(ds, eng, treatment, results)
                except Exception as e:
                    print(f"     FAILED: {type(e).__name__}: {e}")

    print(f"\nDone. Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
