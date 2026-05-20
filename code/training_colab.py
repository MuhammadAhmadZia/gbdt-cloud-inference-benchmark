# ==============================================================================
# PAPER 2 — TRAINING PIPELINE (Run on Google Colab)
# 
# What this script does:
#   1. Downloads and engineers features from two urban datasets
#      - Olist Brazilian E-Commerce → on-time delivery classification
#      - NYC Yellow Taxi            → trip duration regression
#   2. Trains LightGBM, XGBoost, CatBoost on each dataset
#   3. Saves trained models + test sets to Google Drive
#
# After this runs once, you NEVER train again.
# Every inference platform loads these saved files.
# ==============================================================================


# ==============================================================================
# CELL 1 — Install dependencies
# ==============================================================================
# Run this cell first. Restart runtime after if prompted.

# !pip install lightgbm xgboost catboost scikit-learn pandas numpy pyarrow -q


# ==============================================================================
# CELL 2 — Imports and folder structure
# ==============================================================================

import pandas as pd
import numpy as np
import os, json, pickle, warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             mean_squared_error, r2_score)
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings('ignore')

# ── Directory structure ────────────────────────────────────────────────────────
# Mount Drive so models persist after Colab session ends
from google.colab import drive
drive.mount('/content/drive')

BASE_DIR = '/content/drive/MyDrive/paper2_benchmark'

for d in [
    f'{BASE_DIR}/saved_models/olist',
    f'{BASE_DIR}/saved_models/nyc_taxi',
    f'{BASE_DIR}/test_data',
    f'{BASE_DIR}/metadata',
]:
    os.makedirs(d, exist_ok=True)

print("Folder structure ready.")


# ==============================================================================
# CELL 3 — Download Olist dataset
# ==============================================================================
# Requires Kaggle API credentials. 
# Upload your kaggle.json when prompted, OR manually download the zip from 
# https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
# and upload it to Colab's file browser, then skip the API lines.

# --- Option A: Kaggle API (recommended) ---
# !pip install kaggle -q
# from google.colab import files
# print("Upload your kaggle.json file:")
# files.upload()
# !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d olistbr/brazilian-ecommerce -p /content/olist_raw/ --unzip -q
# print("Olist downloaded.")

# --- Option B: If you uploaded the zip manually ---
# !unzip -q /content/brazilian-ecommerce.zip -d /content/olist_raw/

OLIST_DIR = '/content/olist_raw'


# ==============================================================================
# CELL 4 — Olist feature engineering
# ==============================================================================

def build_olist_features(olist_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merges Olist tables and engineers features.
    
    Target: is_late (1 = delivered after estimated date, 0 = on time)
    This maps directly to the sustainability framing:
    late deliveries → re-delivery attempts → extra vehicle kilometres → emissions.
    
    Returns: (X, y) ready for train/test split
    """
    print("Loading Olist tables...")

    orders   = pd.read_csv(f'{olist_dir}/olist_orders_dataset.csv',
                           parse_dates=['order_purchase_timestamp',
                                        'order_approved_at',
                                        'order_delivered_carrier_date',
                                        'order_delivered_customer_date',
                                        'order_estimated_delivery_date'])
    items    = pd.read_csv(f'{olist_dir}/olist_order_items_dataset.csv')
    products = pd.read_csv(f'{olist_dir}/olist_products_dataset.csv')
    customers= pd.read_csv(f'{olist_dir}/olist_customers_dataset.csv')
    sellers  = pd.read_csv(f'{olist_dir}/olist_sellers_dataset.csv')

    # ── Keep only delivered orders ─────────────────────────────────────────────
    df = orders[orders['order_status'] == 'delivered'].copy()
    df = df.dropna(subset=['order_delivered_customer_date',
                            'order_estimated_delivery_date'])

    # ── Target variable ────────────────────────────────────────────────────────
    df['is_late'] = (
        df['order_delivered_customer_date'] > df['order_estimated_delivery_date']
    ).astype(int)

    print(f"  Class balance — on-time: {(df['is_late']==0).sum():,} "
          f"| late: {(df['is_late']==1).sum():,}")

    # ── Order-level time features ──────────────────────────────────────────────
    df['purchase_hour']     = df['order_purchase_timestamp'].dt.hour
    df['purchase_dayofweek']= df['order_purchase_timestamp'].dt.dayofweek
    df['purchase_month']    = df['order_purchase_timestamp'].dt.month
    df['estimated_days']    = (
        df['order_estimated_delivery_date'] - df['order_purchase_timestamp']
    ).dt.days

    # ── Aggregate item-level features per order ────────────────────────────────
    item_agg = items.groupby('order_id').agg(
        n_items       = ('order_item_id', 'count'),
        total_price   = ('price', 'sum'),
        total_freight = ('freight_value', 'sum'),
        mean_price    = ('price', 'mean'),
    ).reset_index()

    # ── Product features (merge via items) ────────────────────────────────────
    items_prod = items.merge(
        products[['product_id','product_weight_g',
                  'product_length_cm','product_height_cm','product_width_cm',
                  'product_category_name']],
        on='product_id', how='left'
    )
    items_prod['product_volume_cm3'] = (
        items_prod['product_length_cm'] *
        items_prod['product_height_cm'] *
        items_prod['product_width_cm']
    )
    prod_agg = items_prod.groupby('order_id').agg(
        mean_weight  = ('product_weight_g', 'mean'),
        mean_volume  = ('product_volume_cm3', 'mean'),
        top_category = ('product_category_name', lambda x: x.mode()[0]
                         if len(x) > 0 else 'unknown')
    ).reset_index()

    # ── Seller info: how many unique sellers in the order ─────────────────────
    seller_agg = items.groupby('order_id').agg(
        n_sellers      = ('seller_id', 'nunique'),
    ).reset_index()
    seller_state = items.merge(sellers[['seller_id','seller_state']],
                               on='seller_id', how='left')
    seller_state_agg = seller_state.groupby('order_id').agg(
        seller_state = ('seller_state', lambda x: x.mode()[0]
                         if len(x) > 0 else 'unknown')
    ).reset_index()

    # ── Customer state ─────────────────────────────────────────────────────────
    cust = customers[['customer_id','customer_state']]

    # ── Merge everything ───────────────────────────────────────────────────────
    df = (df
          .merge(item_agg,          on='order_id',   how='left')
          .merge(prod_agg,          on='order_id',   how='left')
          .merge(seller_agg,        on='order_id',   how='left')
          .merge(seller_state_agg,  on='order_id',   how='left')
          .merge(cust,              on='customer_id', how='left')
         )

    # ── Same-state flag (proxy for short delivery distance) ───────────────────
    df['same_state'] = (df['seller_state'] == df['customer_state']).astype(int)

    # ── Encode categoricals ────────────────────────────────────────────────────
    for col in ['seller_state', 'customer_state', 'top_category']:
        df[col] = df[col].fillna('unknown')
        df[col] = df[col].astype('category').cat.codes

    # ── Final feature list ─────────────────────────────────────────────────────
    FEATURES = [
        'purchase_hour', 'purchase_dayofweek', 'purchase_month',
        'estimated_days',
        'n_items', 'total_price', 'total_freight', 'mean_price',
        'mean_weight', 'mean_volume',
        'n_sellers', 'same_state',
        'seller_state', 'customer_state', 'top_category',
    ]
    TARGET = 'is_late'

    df = df[FEATURES + [TARGET]].dropna()

    print(f"  Final dataset shape: {df.shape}")
    return df[FEATURES], df[TARGET]


X_olist, y_olist = build_olist_features(OLIST_DIR)

# Train / test split — stratified because classes are imbalanced
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
    X_olist, y_olist, test_size=0.2, random_state=42, stratify=y_olist
)
print(f"Olist — Train: {X_train_o.shape}, Test: {X_test_o.shape}")

# Save test set (inference platforms will load this, not the full dataset)
X_test_o.to_parquet(f'{BASE_DIR}/test_data/olist_X_test.parquet', index=False)
y_test_o.to_parquet(f'{BASE_DIR}/test_data/olist_y_test.parquet', index=False)
print("Test data saved.")


# ==============================================================================
# CELL 5 — Download NYC Yellow Taxi dataset
# ==============================================================================
# We use January + February + March 2023 (≈9M rows), then sample 600K for speed.
# Direct download from TLC — no Kaggle account needed.

# !wget -q https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet \
#      -O /content/nyc_jan.parquet
# !wget -q https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet \
#      -O /content/nyc_feb.parquet
# !wget -q https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet \
#      -O /content/nyc_mar.parquet
# print("NYC Taxi downloaded.")


# ==============================================================================
# CELL 6 — NYC Taxi feature engineering
# ==============================================================================

def build_nyc_features(file_paths: list) -> tuple[pd.DataFrame, pd.Series]:
    """
    Builds a trip-duration regression dataset from NYC Yellow Taxi records.
    
    Target: log1p(trip_duration_seconds)
      Log-transform reduces skew from outliers and stabilises training.
      We log-transform here; during evaluation we inverse-transform.
    
    Sustainability framing: accurate trip-duration prediction enables better 
    dispatch decisions, reducing deadhead miles and idle time in urban fleets.
    """
    print("Loading NYC Taxi files...")
    chunks = []
    for fp in file_paths:
        chunk = pd.read_parquet(fp)
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f"  Raw rows: {len(df):,}")

    # ── Basic cleaning ─────────────────────────────────────────────────────────
    df['trip_duration_s'] = (
        df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    ).dt.total_seconds()

    # Remove clearly bad rows
    df = df[
        (df['trip_duration_s'] >= 60)   &   # at least 1 minute
        (df['trip_duration_s'] <= 7200) &   # at most 2 hours
        (df['trip_distance']   >  0)    &
        (df['trip_distance']   <= 60)   &   # miles, remove airport outliers
        (df['passenger_count'] >= 1)    &
        (df['passenger_count'] <= 6)
    ].copy()
    print(f"  After cleaning: {len(df):,}")

    # Sample to 600K for training efficiency (still larger than most GBDT papers)
    if len(df) > 600_000:
        df = df.sample(600_000, random_state=42)
    print(f"  After sampling: {len(df):,}")

    # ── Time features ──────────────────────────────────────────────────────────
    df['pickup_hour']      = df['tpep_pickup_datetime'].dt.hour
    df['pickup_dayofweek'] = df['tpep_pickup_datetime'].dt.dayofweek
    df['pickup_month']     = df['tpep_pickup_datetime'].dt.month
    df['is_weekend']       = (df['pickup_dayofweek'] >= 5).astype(int)
    df['is_rush_hour']     = (
        df['pickup_hour'].isin([7, 8, 9, 17, 18, 19])
    ).astype(int)

    # ── Location features ──────────────────────────────────────────────────────
    # PULocationID and DOLocationID are NYC taxi zone IDs (1–263)
    # Same-zone flag: proxy for short intra-neighbourhood trips
    df['same_zone']     = (df['PULocationID'] == df['DOLocationID']).astype(int)
    df['zone_distance'] = (df['PULocationID'] - df['DOLocationID']).abs()

    # ── Rate and payment ───────────────────────────────────────────────────────
    df['RatecodeID']   = df['RatecodeID'].fillna(1).clip(1, 6).astype(int)
    df['payment_type'] = df['payment_type'].fillna(1).clip(1, 5).astype(int)

    # ── Target ────────────────────────────────────────────────────────────────
    df['log_duration'] = np.log1p(df['trip_duration_s'])

    FEATURES = [
        'trip_distance',
        'pickup_hour', 'pickup_dayofweek', 'pickup_month',
        'is_weekend', 'is_rush_hour',
        'PULocationID', 'DOLocationID',
        'same_zone', 'zone_distance',
        'passenger_count',
        'RatecodeID', 'payment_type',
    ]
    TARGET = 'log_duration'

    df = df[FEATURES + [TARGET]].dropna()
    print(f"  Final shape: {df.shape}")
    return df[FEATURES], df[TARGET]


nyc_files = ['/content/nyc_jan.parquet',
             '/content/nyc_feb.parquet',
             '/content/nyc_mar.parquet']

X_nyc, y_nyc = build_nyc_features(nyc_files)

X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
    X_nyc, y_nyc, test_size=0.2, random_state=42
)
print(f"NYC — Train: {X_train_n.shape}, Test: {X_test_n.shape}")

X_test_n.to_parquet(f'{BASE_DIR}/test_data/nyc_X_test.parquet', index=False)
y_test_n.to_parquet(f'{BASE_DIR}/test_data/nyc_y_test.parquet', index=False)
print("NYC test data saved.")


# ==============================================================================
# CELL 7 — Model training helpers
# ==============================================================================

def train_lgbm_classification(X_train, y_train, X_val, y_val):
    params = dict(
        objective        = 'binary',
        metric           = 'auc',
        n_estimators     = 500,
        learning_rate    = 0.05,
        num_leaves       = 63,
        max_depth        = -1,
        min_child_samples= 20,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        n_jobs           = -1,
        random_state     = 42,
        verbose          = -1,
    )
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=-1)])
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    print(f"    LightGBM  AUC: {auc:.4f}  |  trees: {model.n_estimators_}")
    return model

def train_lgbm_regression(X_train, y_train, X_val, y_val):
    params = dict(
        objective        = 'regression',
        metric           = 'rmse',
        n_estimators     = 500,
        learning_rate    = 0.05,
        num_leaves       = 63,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        n_jobs           = -1,
        random_state     = 42,
        verbose          = -1,
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=-1)])
    rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
    print(f"    LightGBM  RMSE: {rmse:.4f}  |  trees: {model.n_estimators_}")
    return model


def train_xgb_classification(X_train, y_train, X_val, y_val):
    model = xgb.XGBClassifier(
        n_estimators     = 500,
        learning_rate    = 0.05,
        max_depth        = 6,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        use_label_encoder= False,
        eval_metric      = 'auc',
        early_stopping_rounds = 50,
        n_jobs           = -1,
        random_state     = 42,
        verbosity        = 0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)], verbose=False)
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    print(f"    XGBoost   AUC: {auc:.4f}  |  trees: {model.best_iteration}")
    return model

def train_xgb_regression(X_train, y_train, X_val, y_val):
    model = xgb.XGBRegressor(
        n_estimators     = 500,
        learning_rate    = 0.05,
        max_depth        = 6,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        eval_metric      = 'rmse',
        early_stopping_rounds = 50,
        n_jobs           = -1,
        random_state     = 42,
        verbosity        = 0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)], verbose=False)
    rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
    print(f"    XGBoost   RMSE: {rmse:.4f}  |  trees: {model.best_iteration}")
    return model


def train_catboost_classification(X_train, y_train, X_val, y_val):
    model = cb.CatBoostClassifier(
        iterations       = 500,
        learning_rate    = 0.05,
        depth            = 6,
        loss_function    = 'Logloss',
        eval_metric      = 'AUC',
        early_stopping_rounds = 50,
        random_seed      = 42,
        verbose          = False,
    )
    model.fit(X_train, y_train,
              eval_set=(X_val, y_val), verbose=False)
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    print(f"    CatBoost  AUC: {auc:.4f}  |  trees: {model.tree_count_}")
    return model

def train_catboost_regression(X_train, y_train, X_val, y_val):
    model = cb.CatBoostRegressor(
        iterations       = 500,
        learning_rate    = 0.05,
        depth            = 6,
        loss_function    = 'RMSE',
        early_stopping_rounds = 50,
        random_seed      = 42,
        verbose          = False,
    )
    model.fit(X_train, y_train,
              eval_set=(X_val, y_val), verbose=False)
    rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
    print(f"    CatBoost  RMSE: {rmse:.4f}  |  trees: {model.tree_count_}")
    return model


# ==============================================================================
# CELL 8 — Train all models: OLIST
# ==============================================================================

print("\n" + "="*60)
print("TRAINING — OLIST (on-time delivery classification)")
print("="*60)

# Use 10% of training data as internal validation for early stopping
X_tr_o, X_val_o, y_tr_o, y_val_o = train_test_split(
    X_train_o, y_train_o, test_size=0.1, random_state=42, stratify=y_train_o
)

print("\n  Training LightGBM...")
lgbm_olist = train_lgbm_classification(X_tr_o, y_tr_o, X_val_o, y_val_o)

print("\n  Training XGBoost...")
xgb_olist = train_xgb_classification(X_tr_o, y_tr_o, X_val_o, y_val_o)

print("\n  Training CatBoost...")
cat_olist = train_catboost_classification(X_tr_o, y_tr_o, X_val_o, y_val_o)

# ── Final evaluation on held-out test set ─────────────────────────────────────
print("\n  Final test-set AUC:")
for name, model in [('LightGBM', lgbm_olist),
                     ('XGBoost',  xgb_olist),
                     ('CatBoost', cat_olist)]:
    proba = model.predict_proba(X_test_o)[:, 1]
    auc   = roc_auc_score(y_test_o, proba)
    print(f"    {name:12s}: {auc:.4f}")

# ── Save Olist models ──────────────────────────────────────────────────────────
lgbm_olist.booster_.save_model(f'{BASE_DIR}/saved_models/olist/lgbm.txt')
xgb_olist.save_model(f'{BASE_DIR}/saved_models/olist/xgb.json')
cat_olist.save_model(f'{BASE_DIR}/saved_models/olist/catboost.cbm')
print("\nOlist models saved.")


# ==============================================================================
# CELL 9 — Train all models: NYC TAXI
# ==============================================================================

print("\n" + "="*60)
print("TRAINING — NYC TAXI (trip duration regression)")
print("="*60)

X_tr_n, X_val_n, y_tr_n, y_val_n = train_test_split(
    X_train_n, y_train_n, test_size=0.1, random_state=42
)

print("\n  Training LightGBM...")
lgbm_nyc = train_lgbm_regression(X_tr_n, y_tr_n, X_val_n, y_val_n)

print("\n  Training XGBoost...")
xgb_nyc = train_xgb_regression(X_tr_n, y_tr_n, X_val_n, y_val_n)

print("\n  Training CatBoost...")
cat_nyc = train_catboost_regression(X_tr_n, y_tr_n, X_val_n, y_val_n)

# ── Final evaluation ───────────────────────────────────────────────────────────
print("\n  Final test-set RMSE (log-seconds):")
for name, model in [('LightGBM', lgbm_nyc),
                     ('XGBoost',  xgb_nyc),
                     ('CatBoost', cat_nyc)]:
    preds = model.predict(X_test_n)
    rmse  = np.sqrt(mean_squared_error(y_test_n, preds))
    r2    = r2_score(y_test_n, preds)
    print(f"    {name:12s}:  RMSE={rmse:.4f}  R²={r2:.4f}")

# ── Save NYC models ────────────────────────────────────────────────────────────
lgbm_nyc.booster_.save_model(f'{BASE_DIR}/saved_models/nyc_taxi/lgbm.txt')
xgb_nyc.save_model(f'{BASE_DIR}/saved_models/nyc_taxi/xgb.json')
cat_nyc.save_model(f'{BASE_DIR}/saved_models/nyc_taxi/catboost.cbm')
print("\nNYC Taxi models saved.")


# ==============================================================================
# CELL 10 — Save metadata (feature names, task types, baseline metrics)
# ==============================================================================
# This metadata travels with the models so the inference harness
# knows what columns to expect and how to evaluate.

metadata = {
    "olist": {
        "task":         "classification",
        "target":       "is_late",
        "n_train":      int(len(X_train_o)),
        "n_test":       int(len(X_test_o)),
        "n_features":   int(X_olist.shape[1]),
        "feature_names": list(X_olist.columns),
        "metric":       "roc_auc",
        "sustainability_framing": (
            "Late delivery prediction for last-mile logistics. "
            "Correct predictions enable route optimisation, "
            "reducing failed deliveries and associated vehicle emissions."
        ),
    },
    "nyc_taxi": {
        "task":         "regression",
        "target":       "log_duration",
        "n_train":      int(len(X_train_n)),
        "n_test":       int(len(X_test_n)),
        "n_features":   int(X_nyc.shape[1]),
        "feature_names": list(X_nyc.columns),
        "metric":       "rmse_log",
        "sustainability_framing": (
            "Trip duration prediction for urban taxi dispatch. "
            "Accurate predictions reduce deadhead miles and idle time "
            "in urban vehicle fleets."
        ),
    },
}

with open(f'{BASE_DIR}/metadata/datasets.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("Metadata saved.")
print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"\nAll files saved to: {BASE_DIR}")
print("\nFiles structure:")
print("  saved_models/olist/     → lgbm.txt, xgb.json, catboost.cbm")
print("  saved_models/nyc_taxi/  → lgbm.txt, xgb.json, catboost.cbm")
print("  test_data/              → olist_X_test.parquet, olist_y_test.parquet")
print("                          → nyc_X_test.parquet, nyc_y_test.parquet")
print("  metadata/               → datasets.json")
print("\nNext step: run paper2_inference_harness.py on each hardware platform.")
