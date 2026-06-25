# ==============================================================================
# GBDT INFERENCE BENCHMARK — DETAILED TIMING AND ENERGY HARNESS
#
# Loads pre-trained GBDT models and benchmarks native-engine prediction across
# precision profiles, recording a full per-call timing decomposition, raw
# per-repetition latencies, input cardinality, and a sustained energy window.
#
# Timing decomposition recorded per configuration:
#   prep_ms        : cost of input casting / quantization / re-expansion
#   xgb_dmatrix_ms : cost of building xgb.DMatrix (XGBoost only)
#   predict_ms     : prediction call on the prepared array (50 raw reps kept)
# Input preparation is measured separately from the prediction call so the two
# are never conflated, and the DMatrix build is reported on its own so the three
# engines are compared on an equal prediction-only basis.
#
# Raw per-rep latencies are retained to support configuration-level
# (hierarchical) statistical modelling of the repeated measurements.
#
# Energy is measured over a sustained fixed-time window rather than single
# millisecond-scale calls, with the CodeCarbon sampling interval and the energy
# source (RAPL vs. load estimate) recorded alongside each measurement.
#
# Input cardinality (mean distinct values per feature column) is recorded to
# characterise how quantization affects tree traversal.
#
# INSTALL:
#   pip install lightgbm xgboost catboost codecarbon scikit-learn pandas numpy pyarrow -q
# ==============================================================================

import os, json, time, platform, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.metrics import roc_auc_score, mean_squared_error
from codecarbon import EmissionsTracker

warnings.filterwarnings('ignore')

# ==============================================================================
# SECTION 1 — PLATFORM SETTINGS (set per environment)
# ==============================================================================

PLATFORM_TAG = "broadwell_colab"

# Colab:       '/content/drive/MyDrive/paper2_benchmark'
# Kaggle:      '/kaggle/input/.../paper2_benchmark'
# Codespaces:  '/workspaces/.../paper2_benchmark'
BASE_DIR = '/content/drive/MyDrive/paper2_benchmark'

# ==============================================================================
# SECTION 2 — CONFIG
# ==============================================================================

N_REPEATS          = 50          # timed repetitions per configuration
BATCH_SIZE         = 10_000      # fixed inference batch (first N test rows)
PREP_REPS          = 10          # reps used to estimate prep / dmatrix cost
SUSTAINED_SECS     = 20          # sustained energy window
MEASURE_POWER_SECS = 1.0         # CodeCarbon sampling interval

OUTPUT_PATH = f'results_detailed_{PLATFORM_TAG}.json'

DATASETS = {
    'olist':    {'task': 'classification', 'metric': 'roc_auc'},
    'nyc_taxi': {'task': 'regression',     'metric': 'rmse'},
}
PRECISIONS = ['fp64', 'fp32', 'fp16', 'bf16', 'int8']

# ==============================================================================
# SECTION 3 — HARDWARE FINGERPRINT
# ==============================================================================

def fingerprint_hardware() -> dict:
    info = {
        'platform_tag':    PLATFORM_TAG,
        'python_platform': platform.platform(),
        'processor':       platform.processor(),
        'machine':         platform.machine(),
        'cpu_count':       os.cpu_count(),
    }
    try:
        with open('/proc/cpuinfo') as f:
            cpuinfo = f.read()
        for line in cpuinfo.splitlines():
            if 'model name' in line.lower():
                info['cpu_model'] = line.split(':')[1].strip(); break
        for line in cpuinfo.splitlines():
            if line.startswith('flags'):
                flags = set(line.split(':')[1].split())
                info['isa_flags'] = {
                    'avx': 'avx' in flags, 'avx2': 'avx2' in flags,
                    'avx512f': 'avx512f' in flags, 'avx512vnni': 'avx512_vnni' in flags,
                    'f16c': 'f16c' in flags, 'fma': 'fma' in flags,
                }
                break
    except Exception as e:
        info['cpuinfo_error'] = str(e)
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if 'MemTotal' in line:
                    info['ram_kb'] = int(line.split()[1]); break
    except Exception:
        pass
    return info

hw = fingerprint_hardware()
print("=" * 60, "\nHARDWARE FINGERPRINT\n", "=" * 60, sep="")
for k, v in hw.items():
    print(f"  {k}: {v}")
print()

# ==============================================================================
# SECTION 4 — LOAD MODELS AND TEST DATA
# ==============================================================================

def load_models(base_dir, dataset, task):
    d = f'{base_dir}/saved_models/{dataset}'
    lgbm_model = lgb.Booster(model_file=f'{d}/lgbm.txt')
    xgb_model  = xgb.XGBClassifier() if task == 'classification' else xgb.XGBRegressor()
    xgb_model.load_model(f'{d}/xgb.json')
    cat_model  = cb.CatBoostClassifier() if task == 'classification' else cb.CatBoostRegressor()
    cat_model.load_model(f'{d}/catboost.cbm')
    return {'lightgbm': lgbm_model, 'xgboost': xgb_model, 'catboost': cat_model}

def load_test_data(base_dir, dataset):
    X = pd.read_parquet(f'{base_dir}/test_data/{dataset}_X_test.parquet')
    y = pd.read_parquet(f'{base_dir}/test_data/{dataset}_y_test.parquet').squeeze()
    return X, y

print("Loading models and test data...")
models_store, test_data_store = {}, {}
for ds, cfg in DATASETS.items():
    models_store[ds]    = load_models(BASE_DIR, ds, cfg['task'])
    test_data_store[ds] = load_test_data(BASE_DIR, ds)
    Xd, yd = test_data_store[ds]
    print(f"  {ds}: X={Xd.shape}, y={yd.shape}")
print("All loaded.\n")

# ==============================================================================
# SECTION 5 — CASTING / PREDICT
# ==============================================================================

def cast_input(X: pd.DataFrame, precision: str) -> pd.DataFrame:
    arr = X.values.astype(np.float64)
    if precision == 'fp64':
        casted = arr.astype(np.float64)
    elif precision == 'fp32':
        casted = arr.astype(np.float32)
    elif precision == 'fp16':
        casted = arr.astype(np.float16).astype(np.float32)
        casted = np.nan_to_num(casted, nan=0.0, posinf=65504.0, neginf=-65504.0)
    elif precision == 'bf16':
        try:
            import torch
            casted = torch.tensor(arr, dtype=torch.bfloat16).to(torch.float32).numpy()
        except ImportError:
            casted = arr.astype(np.float16).astype(np.float32)
            casted = np.nan_to_num(casted, nan=0.0, posinf=65504.0, neginf=-65504.0)
    elif precision == 'int8':
        col_min = arr.min(axis=0, keepdims=True)
        col_max = arr.max(axis=0, keepdims=True)
        denom   = np.where(col_max - col_min == 0, 1, col_max - col_min)
        casted  = ((arr - col_min) / denom * 254 - 127).astype(np.int8).astype(np.float32)
    else:
        raise ValueError(f"Unknown precision: {precision}")
    return pd.DataFrame(casted, columns=X.columns)

def predict_lgb(model, X_df, task):
    return model.predict(X_df.values)

def predict_cat(model, X_df, task):
    if task == 'classification':
        return model.predict_proba(X_df)[:, 1]
    return model.predict(X_df)

def predict_xgb(model, dmatrix, task):
    # DMatrix is prebuilt outside the timed loop; prediction only here.
    return model.get_booster().predict(dmatrix)

def compute_metric(preds, y_true, metric):
    if metric == 'roc_auc':
        try:    return float(roc_auc_score(y_true, preds))
        except Exception: return float('nan')
    if metric == 'rmse':
        return float(np.sqrt(mean_squared_error(y_true, preds)))
    return float('nan')

def time_predict_loop(call_fn):
    """Run 3 warm-ups then N_REPEATS timed reps. Return raw list of ms."""
    for _ in range(3):
        call_fn()
    raw = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        call_fn()
        t1 = time.perf_counter()
        raw.append((t1 - t0) * 1000.0)
    return raw

def measure_energy(call_fn, seconds):
    """Sustained energy over a fixed time window. Returns dict."""
    tracker = EmissionsTracker(output_dir='/tmp', log_level='error',
                               save_to_file=False,
                               measure_power_secs=MEASURE_POWER_SECS)
    tracker.start()
    t_end = time.perf_counter() + seconds
    n = 0
    while time.perf_counter() < t_end:
        call_fn(); n += 1
    tracker.stop()
    try:
        total_kwh = float(tracker._total_energy.kWh)
    except Exception:
        total_kwh = float('nan')
    src = 'unknown'
    try:
        src = 'rapl' if getattr(tracker, '_cpu', None) and \
              tracker._cpu._mode == 'intel_rapl' else 'load_estimate'
    except Exception:
        pass
    return {'sustained_secs': seconds, 'iterations': n,
            'total_kwh': total_kwh,
            'per_inference_kwh': (total_kwh / n) if n else float('nan'),
            'measure_power_secs': MEASURE_POWER_SECS,
            'energy_source': src}

# ==============================================================================
# SECTION 6 — BENCHMARK LOOP
# ==============================================================================

results = {
    'schema_version': 2,
    'hardware': hw,
    'config': {'n_repeats': N_REPEATS, 'batch_size': BATCH_SIZE,
               'prep_reps': PREP_REPS, 'sustained_secs': SUSTAINED_SECS,
               'measure_power_secs': MEASURE_POWER_SECS,
               'platform_tag': PLATFORM_TAG,
               'timing_boundary': ('prep_ms = cast_input(); '
                   'xgb_dmatrix_ms = DMatrix build (xgb only, built once outside loop); '
                   'predict_ms (raw, x50) = model.predict on prepared array only')},
    'experiments': []
}

print("=" * 60, "\nBENCHMARK LOOP\n", "=" * 60, sep="")

for dataset_name, cfg in DATASETS.items():
    task, metric = cfg['task'], cfg['metric']
    X_test, y_test = test_data_store[dataset_name]
    X_batch = X_test.iloc[:BATCH_SIZE].copy()
    y_batch = y_test.iloc[:BATCH_SIZE].copy()

    for model_name, model in models_store[dataset_name].items():
        for precision in PRECISIONS:
            print(f"\n  [{dataset_name}] [{model_name}] [{precision}]")

            # ---- prep cost: time cast_input separately ----
            prep_times = []
            for _ in range(PREP_REPS):
                tp0 = time.perf_counter()
                X_arr = cast_input(X_batch, precision)
                tp1 = time.perf_counter()
                prep_times.append((tp1 - tp0) * 1000.0)
            prep_ms_mean = float(np.mean(prep_times))

            # input cardinality: mean distinct values per column
            try:
                card = float(np.mean([np.unique(X_arr.values[:, j]).size
                                      for j in range(X_arr.shape[1])]))
            except Exception:
                card = float('nan')

            # ---- build engine-specific call; DMatrix outside timing ----
            xgb_dmatrix_ms = None
            if model_name == 'lightgbm':
                call_fn = lambda: predict_lgb(model, X_arr, task)
            elif model_name == 'catboost':
                call_fn = lambda: predict_cat(model, X_arr, task)
            elif model_name == 'xgboost':
                td0 = time.perf_counter()
                dmat = xgb.DMatrix(X_arr)
                td1 = time.perf_counter()
                xgb_dmatrix_ms = (td1 - td0) * 1000.0
                call_fn = lambda: predict_xgb(model, dmat, task)

            # ---- timed predict loop, raw latencies kept ----
            try:
                raw_ms = time_predict_loop(call_fn)
            except Exception as e:
                print(f"    SKIP — predict failed: {e}")
                continue

            preds = call_fn()
            accuracy = compute_metric(preds, y_batch.values, metric)

            energy = measure_energy(call_fn, SUSTAINED_SECS)

            raw_arr = np.array(raw_ms)
            exp = {
                'dataset': dataset_name, 'model': model_name, 'precision': precision,
                'n_reps': len(raw_ms),
                'prep_ms': prep_ms_mean,
                'xgb_dmatrix_ms': xgb_dmatrix_ms,
                'input_mean_cardinality': card,
                'latency_ms': {
                    'raw': raw_ms,
                    'mean': float(raw_arr.mean()),
                    'std': float(raw_arr.std()),
                    'median': float(np.median(raw_arr)),
                    'p95': float(np.percentile(raw_arr, 95)),
                    'p5': float(np.percentile(raw_arr, 5)),
                },
                'energy': energy,
                'accuracy': {'metric': metric, 'value': accuracy},
            }
            results['experiments'].append(exp)
            print(f"    predict: {exp['latency_ms']['mean']:.1f} ± {exp['latency_ms']['std']:.1f} ms"
                  f" | prep: {prep_ms_mean:.2f} ms"
                  + (f" | dmatrix: {xgb_dmatrix_ms:.2f} ms" if xgb_dmatrix_ms else "")
                  + f" | {metric}: {accuracy:.4f}")

with open(OUTPUT_PATH, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nDone. Saved {OUTPUT_PATH}  ({len(results['experiments'])} experiments)")
