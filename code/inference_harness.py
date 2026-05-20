# ==============================================================================
# PAPER 2 — INFERENCE HARNESS
# Run on all platforms. Change only PLATFORM_TAG and BASE_DIR below.
#
# PLATFORM OPTIONS:
#   "broadwell_colab"        → Google Colab        (Intel Broadwell)
#   "skylake_kaggle"         → Kaggle Notebooks    (Intel Skylake)
#   "epyc_codespaces"        → GitHub Codespaces   (AMD EPYC)
#   "huggingfacecpu"        → HuggingFace Space   (Inter Ice Lake)
#
# INSTALL ON EVERY PLATFORM BEFORE RUNNING:
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
# SECTION 1 — CHANGE THESE TWO LINES ON EACH PLATFORM
# ==============================================================================

PLATFORM_TAG = "broadwell_colab"

# BASE_DIR per platform:
#   Colab:       '/content/drive/MyDrive/paper2_benchmark'
#   Kaggle:      '/kaggle/input/datasets/ahmadziauol/paper2-benchmark/paper2_benchmark'
#   Codespaces:  '/workspaces/Paper2-inference-harness-test/paper2-inference/paper2_benchmark'
BASE_DIR = '/content/drive/MyDrive/paper2_benchmark'


# ==============================================================================
# SECTION 2 — CONFIG
# ==============================================================================

N_REPEATS  = 50
BATCH_SIZE = 10_000
OUTPUT_PATH = f'results_{PLATFORM_TAG}.json'

DATASETS = {
    'olist':    {'task': 'classification', 'metric': 'roc_auc'},
    'nyc_taxi': {'task': 'regression',     'metric': 'rmse'},
}


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
                info['cpu_model'] = line.split(':')[1].strip()
                break
        for line in cpuinfo.splitlines():
            if line.startswith('flags'):
                flags = set(line.split(':')[1].split())
                info['isa_flags'] = {
                    'avx':        'avx'         in flags,
                    'avx2':       'avx2'        in flags,
                    'avx512f':    'avx512f'     in flags,
                    'avx512vnni': 'avx512_vnni' in flags,
                    'f16c':       'f16c'        in flags,
                    'fma':        'fma'         in flags,
                }
                break
    except Exception as e:
        info['cpuinfo_error'] = str(e)
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if 'MemTotal' in line:
                    info['ram_kb'] = int(line.split()[1])
                    break
    except Exception:
        pass
    return info


hw = fingerprint_hardware()
print("=" * 60)
print("HARDWARE FINGERPRINT")
print("=" * 60)
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
models_store    = {}
test_data_store = {}
for ds, cfg in DATASETS.items():
    models_store[ds]    = load_models(BASE_DIR, ds, cfg['task'])
    test_data_store[ds] = load_test_data(BASE_DIR, ds)
    X_ds, y_ds = test_data_store[ds]
    print(f"  {ds}: X={X_ds.shape}, y={y_ds.shape}")
print("All loaded.\n")


# ==============================================================================
# SECTION 5 — PRECISION CASTING AND PREDICT
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


def predict(model_name, model, X_df: pd.DataFrame, task):
    if model_name == 'lightgbm':
        return model.predict(X_df.values)
    elif model_name == 'xgboost':
        dm = xgb.DMatrix(X_df)
        return model.get_booster().predict(dm)
    elif model_name == 'catboost':
        if task == 'classification':
            return model.predict_proba(X_df)[:, 1]
        return model.predict(X_df)


def compute_metric(preds, y_true, metric):
    if metric == 'roc_auc':
        try:
            return float(roc_auc_score(y_true, preds))
        except Exception:
            return float('nan')
    elif metric == 'rmse':
        return float(np.sqrt(mean_squared_error(y_true, preds)))
    return float('nan')


# ==============================================================================
# SECTION 6 — BENCHMARK LOOP
# ==============================================================================

PRECISIONS = ['fp64', 'fp32', 'fp16', 'bf16', 'int8']

results = {
    'hardware':    hw,
    'config': {
        'n_repeats':    N_REPEATS,
        'batch_size':   BATCH_SIZE,
        'platform_tag': PLATFORM_TAG,
    },
    'experiments': []
}

print("=" * 60)
print("BENCHMARK LOOP")
print("=" * 60)

for dataset_name, cfg in DATASETS.items():
    task, metric = cfg['task'], cfg['metric']
    X_test, y_test = test_data_store[dataset_name]
    X_batch = X_test.iloc[:BATCH_SIZE].copy()
    y_batch = y_test.iloc[:BATCH_SIZE].copy()

    for model_name, model in models_store[dataset_name].items():
        for precision in PRECISIONS:
            print(f"\n  [{dataset_name}] [{model_name}] [{precision}]")

            try:
                X_arr = cast_input(X_batch, precision)
            except Exception as e:
                print(f"    SKIP — cast failed: {e}")
                continue

            try:
                for _ in range(3):
                    predict(model_name, model, X_arr, task)
            except Exception as e:
                print(f"    SKIP — warm-up failed: {e}")
                continue

            latencies_ms = []
            preds_last   = None

            tracker = EmissionsTracker(
                output_dir='/tmp',
                log_level='error',
                save_to_file=False,
            )
            tracker.start()

            for rep in range(N_REPEATS):
                t0 = time.perf_counter()
                try:
                    preds = predict(model_name, model, X_arr, task)
                except Exception as e:
                    print(f"    rep {rep} failed: {e}")
                    continue
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000)
                if rep == N_REPEATS - 1:
                    preds_last = preds

            tracker.stop()

            if not latencies_ms:
                print("    All reps failed — skipping.")
                continue

            try:
                total_energy_kwh = tracker._total_energy.kWh
                per_rep_energy   = total_energy_kwh / len(latencies_ms)
            except Exception:
                total_energy_kwh = float('nan')
                per_rep_energy   = float('nan')

            accuracy = compute_metric(preds_last, y_batch.values, metric)

            exp_record = {
                'dataset':   dataset_name,
                'model':     model_name,
                'precision': precision,
                'n_reps':    len(latencies_ms),
                'latency_ms': {
                    'mean':   float(np.mean(latencies_ms)),
                    'std':    float(np.std(latencies_ms)),
                    'median': float(np.median(latencies_ms)),
                    'p95':    float(np.percentile(latencies_ms, 95)),
                    'p5':     float(np.percentile(latencies_ms, 5)),
                },
                'energy_kwh': {
                    'total':   total_energy_kwh,
                    'per_rep': per_rep_energy,
                },
                'accuracy': {
                    'metric': metric,
                    'value':  accuracy,
                },
            }
            results['experiments'].append(exp_record)

            print(f"    latency: {exp_record['latency_ms']['mean']:.1f} ± "
                  f"{exp_record['latency_ms']['std']:.1f} ms "
                  f"| energy/rep: {per_rep_energy*1e6:.2f} µWh "
                  f"| {metric}: {accuracy:.4f}")

with open(OUTPUT_PATH, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nDone. Results saved to {OUTPUT_PATH}")
print(f"Total experiments: {len(results['experiments'])}")
