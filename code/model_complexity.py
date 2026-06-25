# ==============================================================================
# MODEL COMPLEXITY REPORT
#
# Reports the final complexity of each trained GBDT model: number of trees,
# total and per-tree leaves, best iteration, on-disk size, and held-out
# predictive performance. Because early stopping leaves each engine at its own
# optimum, model sizes differ across engines; this report documents those
# differences so latency comparisons can be read against model complexity.
#
# Reads the saved model files only — no inference benchmarking is performed.
#
# INPUT FILES:
#   {BASE_DIR}/saved_models/{dataset}/lgbm.txt
#   {BASE_DIR}/saved_models/{dataset}/xgb.json
#   {BASE_DIR}/saved_models/{dataset}/catboost.cbm
#   {BASE_DIR}/test_data/{dataset}_X_test.parquet   (optional, for performance)
#   {BASE_DIR}/test_data/{dataset}_y_test.parquet   (optional)
#
# INSTALL:
#   pip install lightgbm xgboost catboost pandas numpy scikit-learn pyarrow tabulate -q
# ==============================================================================

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.metrics import roc_auc_score, mean_squared_error

BASE_DIR = '/content/drive/MyDrive/paper2_benchmark'
COMPUTE_PERFORMANCE = True

DATASETS = {'olist': 'classification', 'nyc_taxi': 'regression'}

def file_kb(path):
    try:    return round(os.path.getsize(path) / 1024.0, 1)
    except Exception: return None

def lgb_stats(path):
    b = lgb.Booster(model_file=path)
    df = b.trees_to_dataframe()
    n_trees = int(b.num_trees())
    leaves = int((df['decision_type'].isna()).sum()) if 'decision_type' in df else None
    avg_leaves = round(leaves / n_trees, 1) if (leaves and n_trees) else None
    return {'trees': n_trees, 'total_leaves': leaves, 'avg_leaves_per_tree': avg_leaves,
            'size_kb': file_kb(path), 'model': b}

def xgb_stats(path, task):
    m = xgb.XGBClassifier() if task == 'classification' else xgb.XGBRegressor()
    m.load_model(path)
    booster = m.get_booster()
    dump = booster.get_dump()
    n_trees = len(dump)
    total_leaves = sum(d.count('leaf=') for d in dump)
    best_it = getattr(m, 'best_iteration', None)
    return {'trees': n_trees, 'total_leaves': total_leaves,
            'avg_leaves_per_tree': round(total_leaves / n_trees, 1) if n_trees else None,
            'best_iteration': int(best_it) if best_it is not None else None,
            'size_kb': file_kb(path), 'model': m}

def cat_stats(path, task):
    m = cb.CatBoostClassifier() if task == 'classification' else cb.CatBoostRegressor()
    m.load_model(path)
    n_trees = int(m.tree_count_)
    return {'trees': n_trees, 'total_leaves': None, 'avg_leaves_per_tree': None,
            'best_iteration': n_trees, 'size_kb': file_kb(path), 'model': m}

def perf(model_name, m, X, y, task):
    if task == 'classification':
        if model_name == 'lightgbm':
            p = m.predict(X.values)
        else:
            p = m.predict_proba(X)[:, 1]
        return ('ROC-AUC', round(float(roc_auc_score(y, p)), 4))
    else:
        if model_name == 'lightgbm':
            p = m.predict(X.values)
        else:
            p = m.predict(X)
        return ('RMSE', round(float(np.sqrt(mean_squared_error(y, p))), 4))

rows = []
for ds, task in DATASETS.items():
    d = f'{BASE_DIR}/saved_models/{ds}'
    stats = {
        'lightgbm': lgb_stats(f'{d}/lgbm.txt'),
        'xgboost':  xgb_stats(f'{d}/xgb.json', task),
        'catboost': cat_stats(f'{d}/catboost.cbm', task),
    }
    X = y = None
    if COMPUTE_PERFORMANCE:
        try:
            X = pd.read_parquet(f'{BASE_DIR}/test_data/{ds}_X_test.parquet').iloc[:10000]
            y = pd.read_parquet(f'{BASE_DIR}/test_data/{ds}_y_test.parquet').squeeze().iloc[:10000]
        except Exception as e:
            print(f"  (skipping performance for {ds}: {e})")

    for eng, s in stats.items():
        metric_name, metric_val = (None, None)
        if X is not None:
            metric_name, metric_val = perf(eng, s['model'], X, y, task)
        rows.append({
            'dataset': ds, 'engine': eng,
            'trees': s['trees'],
            'total_leaves': s.get('total_leaves'),
            'avg_leaves_per_tree': s.get('avg_leaves_per_tree'),
            'best_iteration': s.get('best_iteration', s['trees']),
            'model_size_kb': s['size_kb'],
            'metric': metric_name, 'performance': metric_val,
        })

df = pd.DataFrame(rows)
df.to_csv(f'{BASE_DIR}/model_complexity.csv', index=False)
print(f"\nSaved {BASE_DIR}/model_complexity.csv\n")
print(df.to_markdown(index=False))
