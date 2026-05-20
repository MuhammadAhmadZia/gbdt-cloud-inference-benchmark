# GBDT Cloud Inference Benchmark

Code and results for the paper:

> [Author Names], "Energy and Precision Profiling of Gradient-Boosted Decision Tree Inference Across Heterogeneous Cloud CPU Environments for Sustainable Urban Services," *[Conference Name]*, 2026.

## Contents

```
code/      — training, inference, and aggregation scripts
results/   — raw JSON outputs per platform, combined CSV, statistical tests
```

## Reproducing the experiments

Install dependencies:
```bash
pip install lightgbm xgboost catboost scikit-learn pandas numpy pyarrow codecarbon matplotlib scipy
```

Run the three scripts in order:
```bash
python code/training_colab.py        # train once on a reference machine
python code/inference_harness.py     # run on each target platform (edit PLATFORM_TAG and BASE_DIR)
python code/aggregate_results.py     # combine results into figures and tables
```

## Datasets

Raw data not included. Download from the original sources:

- **Olist Brazilian E-Commerce:** https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- **NYC Yellow Taxi Trip Records (Q1 2023):** https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

## Supplementary material

`results/pairwise_significance.txt` contains pairwise Welch's t-test results comparing FP64 against each lower-precision profile (FP32, FP16, BF16, INT8) for every model × platform × dataset combination. This table is referenced in the paper but not included due to page limits.

