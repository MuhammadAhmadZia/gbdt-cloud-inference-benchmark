"""
exp4_real_runtime_benchmark.py

Benchmarks two real optimized inference runtimes against the native engines on
the same models and test data used in the main paper:

  - Treelite (compiled C shared library; supports LightGBM and XGBoost)
  - ONNX Runtime (cross-platform; supports all three engines)
  - native_fp32 baseline (the engines' own predict())

This experiment closes the "real quantized / optimized inference baseline" gap
raised in review. It produces a JSON in the same schema as results_*.json so
the existing analysis code can ingest it without changes.

Designed to run on Google Colab with the project mounted at
  /content/drive/MyDrive/paper2_benchmark/

Robustness features:
  - Smoke test on one config before the full sweep
  - Per-config try / except so one failure does not break the run
  - Incremental save after every successful config (resumable)
  - Skips already-completed configs on re-run
  - No fragile model-text rebuilds (this is what broke threshold_quant)

Threading: all runtimes use their default thread settings (all available cores),
matching the main benchmark for a fair comparison.
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st


# ============================================================
# CONFIG
# ============================================================
BASE_DIR     = Path("/content/drive/MyDrive/paper2_benchmark")
RESULTS_FILE = Path("/content/results_runtime_colab.json")   # rename suffix per platform
WORK_DIR     = Path("/content/runtime_work")
BATCH_SIZE   = 10_000     # match main benchmark
WARMUP_RUNS  = 3
TIMED_RUNS   = 50
PLATFORM_TAG = "colab"    # change per platform when reusing the script


# ============================================================
# COLAB DRIVE (no-op off Colab)
# ============================================================
try:
    from google.colab import drive  # type: ignore
    drive.mount('/content/drive', force_remount=False)
except Exception:
    pass


# ============================================================
# INSTALL MISSING PACKAGES
# ============================================================
def _ensure_packages():
    import importlib
    needed = [
        ("onnxruntime",  "onnxruntime"),
        ("onnxmltools",  "onnxmltools"),
        ("skl2onnx",     "skl2onnx"),
        ("treelite",     "treelite"),
        ("tl2cgen",      "tl2cgen"),
    ]
    missing = []
    for mod, pkg in needed:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[setup] installing: {missing}")
        os.system(f"{sys.executable} -m pip install --quiet " + " ".join(missing))

_ensure_packages()

# Heavy imports after install
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
import onnxruntime as ort
from onnxmltools import convert_lightgbm, convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import treelite
import tl2cgen
from sklearn.metrics import roc_auc_score, mean_squared_error

print(f"[setup] lightgbm {lgb.__version__}   xgboost {xgb.__version__}")
print(f"[setup] onnxruntime {ort.__version__}   treelite {treelite.__version__}   tl2cgen {tl2cgen.__version__}")

WORK_DIR.mkdir(exist_ok=True, parents=True)


# ============================================================
# DATA AND MODEL LOADING
# ============================================================
def task_of(dataset):
    return "classification" if dataset == "olist" else "regression"


def load_test_data(dataset):
    X_df = pd.read_parquet(BASE_DIR / "test_data" / f"{dataset}_X_test.parquet")
    y    = pd.read_parquet(BASE_DIR / "test_data" / f"{dataset}_y_test.parquet").values.ravel()
    X_df = X_df.iloc[:BATCH_SIZE].copy()
    y    = y[:BATCH_SIZE]
    X    = X_df.values.astype(np.float32)
    return X_df, X, y


def load_native_model(dataset, engine):
    mdir = BASE_DIR / "saved_models" / dataset
    if engine == "lightgbm":
        return lgb.Booster(model_file=str(mdir / "lgbm.txt"))
    if engine == "xgboost":
        b = xgb.Booster()
        b.load_model(str(mdir / "xgb.json"))
        return b
    if engine == "catboost":
        m = CatBoostClassifier() if task_of(dataset) == "classification" else CatBoostRegressor()
        m.load_model(str(mdir / "catboost.cbm"))
        return m
    raise ValueError(engine)


# ============================================================
# RUNTIME CONVERSION
# ============================================================
def to_onnx(engine, native_model, dataset, n_features):
    out_path = WORK_DIR / f"{engine}_{dataset}.onnx"
    if engine == "lightgbm":
        initial = [("input", FloatTensorType([None, n_features]))]
        m = convert_lightgbm(native_model, initial_types=initial, target_opset=12)
        out_path.write_bytes(m.SerializeToString())
    elif engine == "xgboost":
        initial = [("input", FloatTensorType([None, n_features]))]
        m = convert_xgboost(native_model, initial_types=initial, target_opset=12)
        out_path.write_bytes(m.SerializeToString())
    elif engine == "catboost":
        native_model.save_model(
            str(out_path),
            format="onnx",
            export_parameters={
                "onnx_domain":         "ai.catboost",
                "onnx_model_version":  1,
                "onnx_graph_name":     f"CatBoost_{dataset}",
            },
        )
    return out_path


def to_treelite(engine, dataset):
    mdir = BASE_DIR / "saved_models" / dataset
    lib  = WORK_DIR / f"{engine}_{dataset}.so"
    if engine == "lightgbm":
        tl_model = treelite.frontend.load_lightgbm_model(str(mdir / "lgbm.txt"))
    elif engine == "xgboost":
        tl_model = treelite.frontend.load_xgboost_model(str(mdir / "xgb.json"))
    else:
        raise NotImplementedError(f"Treelite does not support {engine}")
    tl2cgen.export_lib(tl_model, toolchain="gcc", libpath=str(lib), verbose=False)
    return lib


# ============================================================
# PREDICTION WRAPPERS
# ============================================================
def predict_native(engine, model, X_df, X):
    if engine == "lightgbm":
        return model.predict(X)
    if engine == "xgboost":
        return model.predict(xgb.DMatrix(X_df))
    if engine == "catboost":
        # for classification, return probability of positive class
        return model.predict_proba(X)[:, 1] if isinstance(model, CatBoostClassifier) else model.predict(X)


def predict_onnx(sess, X, task):
    feed = {sess.get_inputs()[0].name: X.astype(np.float32)}
    out  = sess.run(None, feed)
    if task == "classification":
        # outputs typically: [label, probabilities]
        if len(out) >= 2:
            probs = out[1]
            if isinstance(probs, list) and len(probs) > 0 and isinstance(probs[0], dict):
                # zipmap dict form
                return np.array([p[max(p.keys())] for p in probs], dtype=np.float64)
            probs = np.asarray(probs)
            if probs.ndim == 2 and probs.shape[1] >= 2:
                return probs[:, -1]
            return probs.ravel()
        return np.asarray(out[0]).ravel()
    return np.asarray(out[0]).ravel()


def predict_treelite(predictor, X, task):
    dmat = tl2cgen.DMatrix(X.astype(np.float32))
    pred = np.asarray(predictor.predict(dmat))
    while pred.ndim > 2:
        pred = pred.squeeze(axis=-1) if pred.shape[-1] == 1 else pred.squeeze()
    if task == "classification":
        if pred.ndim == 2 and pred.shape[1] == 2:
            return pred[:, 1]
        return pred.ravel()
    return pred.ravel()


# ============================================================
# METRIC AND TIMING
# ============================================================
def score(dataset, y_true, y_pred):
    y_pred = np.asarray(y_pred).ravel()
    if task_of(dataset) == "classification":
        return {"metric": "roc_auc", "value": float(roc_auc_score(y_true, y_pred))}
    return {"metric": "rmse", "value": float(np.sqrt(mean_squared_error(y_true, y_pred)))}


def time_calls(pred_fn, n_warmup=WARMUP_RUNS, n_timed=TIMED_RUNS):
    for _ in range(n_warmup):
        pred_fn()
    samples = np.empty(n_timed, dtype=np.float64)
    for i in range(n_timed):
        t0 = time.perf_counter()
        pred_fn()
        samples[i] = (time.perf_counter() - t0) * 1000.0     # ms
    mean = float(samples.mean())
    std  = float(samples.std(ddof=1))
    sem  = std / np.sqrt(n_timed)
    half = float(st.t.ppf(0.975, df=n_timed - 1) * sem)
    return {
        "n_reps":        int(n_timed),
        "mean_ms":       mean,
        "std_ms":        std,
        "ci95_half_ms":  half,
        "samples_ms":    samples.tolist(),
    }


# ============================================================
# RESULTS FILE
# ============================================================
def load_results():
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {"platform": PLATFORM_TAG, "experiments": []}


def save_results(data):
    RESULTS_FILE.write_text(json.dumps(data, indent=2))


def already_done(data, dataset, engine, runtime):
    return any(
        e["dataset"]   == dataset and
        e["model"]     == engine  and
        e["precision"] == runtime
        for e in data["experiments"]
    )


# ============================================================
# ONE CONFIG
# ============================================================
def run_one(dataset, engine, runtime, results, force=False):
    tag = f"{dataset}/{engine}/{runtime}"
    if (not force) and already_done(results, dataset, engine, runtime):
        print(f"[skip] {tag} (already in results file)")
        return
    print(f"[run ] {tag}")

    try:
        X_df, X, y = load_test_data(dataset)
        n_features = X.shape[1]
        task       = task_of(dataset)

        if runtime == "native_fp32":
            model   = load_native_model(dataset, engine)
            pred_fn = lambda: predict_native(engine, model, X_df, X)

        elif runtime == "onnx_runtime":
            model = load_native_model(dataset, engine)
            onnx_path = to_onnx(engine, model, dataset, n_features)
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            pred_fn = lambda: predict_onnx(sess, X, task)

        elif runtime == "treelite":
            if engine == "catboost":
                print(f"       skip: Treelite does not support catboost")
                return
            lib_path  = to_treelite(engine, dataset)
            predictor = tl2cgen.Predictor(libpath=str(lib_path))
            pred_fn   = lambda: predict_treelite(predictor, X, task)

        else:
            raise ValueError(f"unknown runtime: {runtime}")

        # one prediction call for the accuracy / fidelity check
        y_pred = pred_fn()
        acc    = score(dataset, y, y_pred)

        # 3 warmup + 50 timed
        timing = time_calls(pred_fn)

        exp = {
            "dataset":   dataset,
            "model":     engine,
            "precision": runtime,                      # reusing 'precision' field for runtime tag
            "n_reps":    timing["n_reps"],
            "latency_ms": {
                "mean":         timing["mean_ms"],
                "std":          timing["std_ms"],
                "ci95_half":    timing["ci95_half_ms"],
                "samples":      timing["samples_ms"],
            },
            "accuracy":  acc,
        }
        results["experiments"].append(exp)
        save_results(results)

        print(f"       latency {timing['mean_ms']:7.2f} +/- {timing['ci95_half_ms']:.2f} ms   "
              f"{acc['metric']}={acc['value']:.4f}")

    except Exception as e:
        print(f"       FAILED: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)


# ============================================================
# SMOKE TEST
# ============================================================
def smoke_test():
    print("=" * 64)
    print("SMOKE TEST: lightgbm + olist + onnx_runtime")
    print("=" * 64)
    tmp = {"platform": PLATFORM_TAG, "experiments": []}
    run_one("olist", "lightgbm", "onnx_runtime", tmp, force=True)
    if not tmp["experiments"]:
        raise RuntimeError("Smoke test failed: no result recorded. Fix the error above before continuing.")
    print("SMOKE TEST PASSED\n")


# ============================================================
# SUMMARY
# ============================================================
def print_summary(results):
    rows = []
    for e in results["experiments"]:
        rows.append({
            "dataset":   e["dataset"],
            "engine":    e["model"],
            "runtime":   e["precision"],
            "latency":   e["latency_ms"]["mean"],
            "ci_half":   e["latency_ms"]["ci95_half"],
            "metric":    e["accuracy"]["metric"],
            "score":     e["accuracy"]["value"],
        })
    if not rows:
        print("(no results)")
        return
    df = pd.DataFrame(rows)
    print("\n=== Summary ===")
    print(df.to_string(index=False))
    # speedup table vs native_fp32
    pivot = df.pivot_table(index=["dataset", "engine"], columns="runtime", values="latency")
    print("\n=== Latency (ms) by runtime ===")
    print(pivot.to_string())
    if "native_fp32" in pivot.columns:
        print("\n=== Speedup vs native_fp32 ===")
        sp = pivot.div(pivot["native_fp32"], axis=0).rpow(1).rdiv(1)
        # cleaner: speedup = native / runtime
        sp = pivot["native_fp32"].to_frame().join(pivot.drop(columns=["native_fp32"]))
        for col in sp.columns:
            if col == "native_fp32":
                continue
            sp[col] = sp["native_fp32"] / sp[col]
        sp = sp.drop(columns=["native_fp32"])
        print(sp.round(2).to_string())


# ============================================================
# MAIN
# ============================================================
def main():
    smoke_test()

    results  = load_results()
    datasets = ["olist", "nyc_taxi"]
    engines  = ["lightgbm", "xgboost", "catboost"]
    runtimes = ["native_fp32", "onnx_runtime", "treelite"]

    total = len(datasets) * len(engines) * len(runtimes)
    print("=" * 64)
    print(f"FULL SWEEP: {total} configs (some will be skipped: Treelite + catboost)")
    print("=" * 64)

    for ds in datasets:
        for eng in engines:
            for rt in runtimes:
                run_one(ds, eng, rt, results)

    print("\nDone.")
    print(f"Results file: {RESULTS_FILE}")
    print(f"Configs completed: {len(results['experiments'])}")
    print_summary(results)


if __name__ == "__main__":
    main()
