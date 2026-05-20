"""
PAPER 2 — AGGREGATION v2 (with statistical analysis)
=====================================================
Adds 95% confidence intervals, error bars on all figures,
and a new latency distribution figure addressing reviewer concerns.

Run:
    pip install matplotlib pandas numpy scipy -q
    python paper2_aggregate_v2.py

Generates:
    - results_combined_v2.csv             : full table with CIs
    - fig1_latency_heatmap_v2.png         : heatmap with annotated CI half-width
    - fig2_energy_heatmap_v2.png          : energy heatmap
    - fig3_accuracy_drop_v2.png           : accuracy degradation
    - fig4_int8_speedup_v2.png            : INT8 speedup
    - fig5_fp32_with_errorbars.png        : FP32 cross-platform with 95% CI error bars
    - fig6_latency_distribution.png       : NEW — distribution range plot per platform
    - table_latency_with_ci.txt           : Table with mean ± 95% CI
    - table_pairwise_significance.txt     : Pairwise t-test results (FP64 vs others)
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams['font.family']    = 'DejaVu Sans'
plt.rcParams['font.size']      = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10

PLATFORM_LABELS = {
    'broadwell_colab':    'Colab (Intel Xeon, 2-core, no AVX-512)',
    'skylake_kaggleb':    'Kaggle (Intel Xeon, 4-core, no AVX-512)',
    'epyc_codespaces':    'Codespaces (AMD EPYC 7763, 2-core)',
    'huggingface_cpu':    'HuggingFace (Intel Ice Lake, 16-core, VNNI)',
}
PLATFORM_SHORT = {
    'broadwell_colab':    'Colab',
    'skylake_kaggleb':    'Kaggle',
    'epyc_codespaces':    'Codespaces',
    'huggingface_cpu':    'HuggingFace',
}
PLATFORM_ORDER = list(PLATFORM_LABELS.keys())

PRECISION_ORDER  = ['fp64', 'fp32', 'fp16', 'bf16', 'int8']
PRECISION_LABELS = ['FP64', 'FP32', 'FP16', 'BF16', 'INT8']
MODEL_ORDER      = ['lightgbm', 'xgboost', 'catboost']
MODEL_LABELS     = ['LightGBM', 'XGBoost', 'CatBoost']
DATASET_ORDER    = ['olist', 'nyc_taxi']
DATASET_LABELS   = ['Olist (Classification)', 'NYC Taxi (Regression)']

FILES = {
    'broadwell_colab':  'results_broadwell_colab.json',
    'skylake_kaggleb':  'results_skylake_kaggleb.json',
    'epyc_codespaces':  'results_epyc_codespaces.json',
    'huggingface_cpu':  'results_huggingface_cpu.json',
}

PLATFORM_COLORS = ['#5C6BC0', '#26A69A', '#EF5350', '#FFA726']

# ── Statistical helpers ────────────────────────────────────────────────────────
def ci_halfwidth(std, n=50, confidence=0.95):
    """95% confidence interval half-width assuming normal distribution."""
    # For n=50 we use z=1.96; for stricter t-distribution it's 2.01
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n-1)
    return t_crit * std / np.sqrt(n)


# ==============================================================================
# SECTION 1 — LOAD AND FLATTEN
# ==============================================================================

records = []
hardware_info = {}

for platform_tag, filename in FILES.items():
    if not os.path.exists(filename):
        print(f"WARNING: {filename} not found, skipping.")
        continue
    with open(filename) as f:
        data = json.load(f)

    hardware_info[platform_tag] = data['hardware']

    for exp in data['experiments']:
        n_reps = exp['n_reps']
        std    = exp['latency_ms']['std']
        ci_half= ci_halfwidth(std, n_reps)

        records.append({
            'platform':      platform_tag,
            'dataset':       exp['dataset'],
            'model':         exp['model'],
            'precision':     exp['precision'],
            'n_reps':        n_reps,
            'latency_mean':  exp['latency_ms']['mean'],
            'latency_std':   std,
            'latency_ci95':  ci_half,
            'latency_p5':    exp['latency_ms']['p5'],
            'latency_p95':   exp['latency_ms']['p95'],
            'latency_median':exp['latency_ms']['median'],
            'energy_per_rep_uwh': exp['energy_kwh']['per_rep'] * 1e6,
            'accuracy_value':     exp['accuracy']['value'],
            'accuracy_metric':    exp['accuracy']['metric'],
        })

df = pd.DataFrame(records)
df.to_csv('results_combined_v2.csv', index=False)
print(f"Loaded {len(df)} experiment records.\n")


# ==============================================================================
# SECTION 2 — TABLE: LATENCY WITH 95% CI
# ==============================================================================

lines = []
lines.append("TABLE: Mean Inference Latency in milliseconds (95% CI half-width in parentheses)")
lines.append("Batch size = 10,000 rows | 50 repetitions per configuration")
lines.append("=" * 100)

for ds in DATASET_ORDER:
    ds_label = DATASET_LABELS[DATASET_ORDER.index(ds)]
    lines.append(f"\nDataset: {ds_label}")
    lines.append(f"{'Model':<10} {'Platform':<14} {'FP64':>14} {'FP32':>14} {'FP16':>14} {'BF16':>14} {'INT8':>14}")
    lines.append("-" * 100)

    for model in MODEL_ORDER:
        for pt in PLATFORM_ORDER:
            sub = df[(df['dataset']==ds) & (df['model']==model) & (df['platform']==pt)]
            if sub.empty:
                continue
            cells = []
            for prec in PRECISION_ORDER:
                row = sub[sub['precision']==prec]
                if row.empty:
                    cells.append("N/A")
                else:
                    m  = row['latency_mean'].values[0]
                    ci = row['latency_ci95'].values[0]
                    cells.append(f"{m:6.1f} (±{ci:4.1f})")
            lines.append(
                f"{MODEL_LABELS[MODEL_ORDER.index(model)]:<10} "
                f"{PLATFORM_SHORT[pt]:<14} "
                f"{cells[0]:>14} {cells[1]:>14} {cells[2]:>14} {cells[3]:>14} {cells[4]:>14}"
            )
        lines.append("")

table_str = '\n'.join(lines)
with open('table_latency_with_ci.txt', 'w') as f:
    f.write(table_str)
print("Latency table with CI saved.")


# ==============================================================================
# SECTION 3 — PAIRWISE SIGNIFICANCE TESTS (FP64 baseline vs each other precision)
# ==============================================================================
# Welch's t-test using mean/std/n. We approximate per-rep distributions as normal,
# which is reasonable for n=50.

def welch_t_from_summary(m1, s1, n1, m2, s2, n2):
    """Welch's t-test from summary statistics."""
    if s1 == 0 and s2 == 0:
        return float('nan'), float('nan')
    se = np.sqrt(s1**2/n1 + s2**2/n2)
    if se == 0:
        return float('nan'), float('nan')
    t = (m1 - m2) / se
    # Welch-Satterthwaite df
    df_num = (s1**2/n1 + s2**2/n2)**2
    df_den = (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)
    df_w = df_num / df_den if df_den > 0 else n1 + n2 - 2
    p   = 2 * (1 - stats.t.cdf(abs(t), df=df_w))
    return t, p

lines = []
lines.append("PAIRWISE SIGNIFICANCE: FP64 baseline vs each other precision (Welch's t-test)")
lines.append("Significant at p < 0.05 marked with *; p < 0.001 marked with ***")
lines.append("=" * 95)

for ds in DATASET_ORDER:
    ds_label = DATASET_LABELS[DATASET_ORDER.index(ds)]
    lines.append(f"\nDataset: {ds_label}")
    lines.append(f"{'Model':<10} {'Platform':<14} {'FP32':>14} {'FP16':>14} {'BF16':>14} {'INT8':>14}")
    lines.append("-" * 95)

    for model in MODEL_ORDER:
        for pt in PLATFORM_ORDER:
            sub = df[(df['dataset']==ds) & (df['model']==model) & (df['platform']==pt)]
            if sub.empty:
                continue
            fp64 = sub[sub['precision']=='fp64']
            if fp64.empty:
                continue
            m1, s1, n1 = fp64['latency_mean'].values[0], fp64['latency_std'].values[0], fp64['n_reps'].values[0]

            cells = []
            for prec in ['fp32', 'fp16', 'bf16', 'int8']:
                row = sub[sub['precision']==prec]
                if row.empty:
                    cells.append("N/A")
                    continue
                m2, s2, n2 = row['latency_mean'].values[0], row['latency_std'].values[0], row['n_reps'].values[0]
                t, p = welch_t_from_summary(m1, s1, n1, m2, s2, n2)
                sig = '***' if p < 0.001 else ('*' if p < 0.05 else 'ns')
                cells.append(f"p={p:.3f} {sig}")

            lines.append(
                f"{MODEL_LABELS[MODEL_ORDER.index(model)]:<10} "
                f"{PLATFORM_SHORT[pt]:<14} "
                f"{cells[0]:>14} {cells[1]:>14} {cells[2]:>14} {cells[3]:>14}"
            )
        lines.append("")

with open('table_pairwise_significance.txt', 'w') as f:
    f.write('\n'.join(lines))
print("Pairwise significance table saved.")


# ==============================================================================
# SECTION 4 — FIGURE 5 (REVISED): FP32 CROSS-PLATFORM WITH 95% CI
# ==============================================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
fig.suptitle('Mean Inference Latency at FP32 with 95% Confidence Intervals\n'
             'Error bars show statistical precision (50 repetitions per configuration)',
             fontsize=12, fontweight='bold')

for ax_idx, (ds, ds_label) in enumerate(zip(DATASET_ORDER, DATASET_LABELS)):
    ax = axes[ax_idx]
    x = np.arange(len(MODEL_ORDER))
    width = 0.18

    for pt_idx, (pt, pt_color) in enumerate(zip(PLATFORM_ORDER, PLATFORM_COLORS)):
        latencies = []
        ci_half   = []
        for model in MODEL_ORDER:
            sub = df[(df['platform']==pt) & (df['model']==model) &
                     (df['dataset']==ds) & (df['precision']=='fp32')]
            latencies.append(sub['latency_mean'].values[0] if not sub.empty else 0)
            ci_half.append(sub['latency_ci95'].values[0]   if not sub.empty else 0)

        offset = (pt_idx - 1.5) * width
        ax.bar(x + offset, latencies, width, yerr=ci_half,
               label=PLATFORM_SHORT[pt],
               color=pt_color, alpha=0.85, edgecolor='white',
               capsize=4, error_kw={'linewidth': 1.2, 'ecolor': '#222'})

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS)
    ax.set_ylabel('Mean Latency (ms) ± 95% CI')
    ax.set_title(ds_label)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('fig5_fp32_with_errorbars.png', dpi=200, bbox_inches='tight')
plt.close()
print("Figure 5 (with CI error bars) saved.")


# ==============================================================================
# SECTION 5 — NEW FIGURE 6: LATENCY DISTRIBUTION RANGE PLOT
# ==============================================================================
# Shows mean (point), 95% CI on mean (thick line), and 5th-95th percentile
# range from the 50 individual runs (thin line). This is the closest to a
# boxplot we can produce from summary statistics.

fig, axes = plt.subplots(2, 1, figsize=(13, 9))
fig.suptitle('Latency Distribution Across Platforms — LightGBM\n'
             'Solid bar: 95% confidence interval on the mean  |  '
             'Thin line: 5th–95th percentile range across 50 individual runs',
             fontsize=12, fontweight='bold')

for ax_idx, (ds, ds_label) in enumerate(zip(DATASET_ORDER, DATASET_LABELS)):
    ax = axes[ax_idx]

    y_pos = 0
    y_ticks = []
    y_labels = []

    for prec_idx, (prec, prec_label) in enumerate(zip(PRECISION_ORDER, PRECISION_LABELS)):
        for pt_idx, (pt, pt_color) in enumerate(zip(PLATFORM_ORDER, PLATFORM_COLORS)):
            sub = df[(df['platform']==pt) & (df['model']=='lightgbm') &
                     (df['dataset']==ds) & (df['precision']==prec)]
            if sub.empty:
                y_pos += 1
                continue
            m   = sub['latency_mean'].values[0]
            ci  = sub['latency_ci95'].values[0]
            p5  = sub['latency_p5'].values[0]
            p95 = sub['latency_p95'].values[0]

            # Thin line: p5-p95 range
            ax.plot([p5, p95], [y_pos, y_pos], color=pt_color,
                    linewidth=1.2, alpha=0.5, zorder=2)
            ax.plot([p5, p5],   [y_pos-0.15, y_pos+0.15],
                    color=pt_color, linewidth=1.2, alpha=0.5, zorder=2)
            ax.plot([p95, p95], [y_pos-0.15, y_pos+0.15],
                    color=pt_color, linewidth=1.2, alpha=0.5, zorder=2)

            # Thick bar: 95% CI on mean
            ax.plot([m-ci, m+ci], [y_pos, y_pos], color=pt_color,
                    linewidth=5.5, alpha=0.95, zorder=3)

            # Mean point
            ax.scatter([m], [y_pos], color='white', s=30,
                       edgecolors=pt_color, linewidths=1.8, zorder=4)

            y_labels.append(f"{prec_label} — {PLATFORM_SHORT[pt]}")
            y_ticks.append(y_pos)
            y_pos += 1
        y_pos += 0.7  # gap between precision groups

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel('Latency (ms)')
    ax.set_title(ds_label)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=c, label=PLATFORM_SHORT[p])
                    for p, c in zip(PLATFORM_ORDER, PLATFORM_COLORS)]
    ax.legend(handles=legend_elems, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('fig6_latency_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print("Figure 6 (distribution range) saved.")


# ==============================================================================
# SECTION 6 — UPDATED FIGURE 1: LATENCY HEATMAP WITH CI ANNOTATIONS
# ==============================================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Mean Inference Latency (ms ± 95% CI half-width)\n'
             'Batch size = 10,000 | 50 repetitions per configuration',
             fontsize=12, fontweight='bold')

for col_idx, (model, model_label) in enumerate(zip(MODEL_ORDER, MODEL_LABELS)):
    for row_idx, (ds, ds_label) in enumerate(zip(DATASET_ORDER, DATASET_LABELS)):
        ax = axes[row_idx][col_idx]

        matrix_mean = np.zeros((len(PLATFORM_ORDER), len(PRECISION_ORDER)))
        matrix_ci   = np.zeros((len(PLATFORM_ORDER), len(PRECISION_ORDER)))

        for i, pt in enumerate(PLATFORM_ORDER):
            for j, prec in enumerate(PRECISION_ORDER):
                sub = df[(df['platform']==pt) & (df['model']==model) &
                         (df['dataset']==ds) & (df['precision']==prec)]
                matrix_mean[i, j] = sub['latency_mean'].values[0] if not sub.empty else np.nan
                matrix_ci[i, j]   = sub['latency_ci95'].values[0]  if not sub.empty else np.nan

        im = ax.imshow(matrix_mean, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(range(len(PRECISION_ORDER)))
        ax.set_xticklabels(PRECISION_LABELS, fontsize=9)
        ax.set_yticks(range(len(PLATFORM_ORDER)))
        ax.set_yticklabels([PLATFORM_SHORT[p] for p in PLATFORM_ORDER], fontsize=8)
        ax.set_title(f'{model_label}\n{ds_label.split("(")[0].strip()}', fontsize=10)

        for i in range(len(PLATFORM_ORDER)):
            for j in range(len(PRECISION_ORDER)):
                m  = matrix_mean[i, j]
                ci = matrix_ci[i, j]
                if not np.isnan(m):
                    ax.text(j, i, f'{m:.0f}\n±{ci:.1f}',
                            ha='center', va='center', fontsize=7,
                            color='black' if m < np.nanmax(matrix_mean)*0.7 else 'white')

        plt.colorbar(im, ax=ax, label='ms')

plt.tight_layout()
plt.savefig('fig1_latency_heatmap_v2.png', dpi=200, bbox_inches='tight')
plt.close()
print("Figure 1 (with CI annotations) saved.")


# ==============================================================================
# SECTION 7 — SUMMARY STATISTICS
# ==============================================================================

print("\n" + "="*70)
print("KEY STATISTICAL FINDINGS")
print("="*70)

# Effect of FP32 vs FP64 across platforms
print("\n[1] FP32 vs FP64 latency change (Olist LightGBM):")
for pt in PLATFORM_ORDER:
    fp64 = df[(df['platform']==pt) & (df['model']=='lightgbm') &
              (df['dataset']=='olist') & (df['precision']=='fp64')]
    fp32 = df[(df['platform']==pt) & (df['model']=='lightgbm') &
              (df['dataset']=='olist') & (df['precision']=='fp32')]
    if fp64.empty or fp32.empty:
        continue
    m1, s1 = fp64['latency_mean'].values[0], fp64['latency_std'].values[0]
    m2, s2 = fp32['latency_mean'].values[0], fp32['latency_std'].values[0]
    t, p = welch_t_from_summary(m1, s1, 50, m2, s2, 50)
    pct = (m1 - m2) / m1 * 100
    print(f"  {PLATFORM_SHORT[pt]:<12}: {pct:+5.1f}% (p={p:.3f})")

# Effect of INT8 vs FP64
print("\n[2] INT8 vs FP64 latency change (Olist LightGBM):")
for pt in PLATFORM_ORDER:
    fp64 = df[(df['platform']==pt) & (df['model']=='lightgbm') &
              (df['dataset']=='olist') & (df['precision']=='fp64')]
    int8 = df[(df['platform']==pt) & (df['model']=='lightgbm') &
              (df['dataset']=='olist') & (df['precision']=='int8')]
    if fp64.empty or int8.empty:
        continue
    m1, s1 = fp64['latency_mean'].values[0], fp64['latency_std'].values[0]
    m2, s2 = int8['latency_mean'].values[0], int8['latency_std'].values[0]
    t, p = welch_t_from_summary(m1, s1, 50, m2, s2, 50)
    pct = (m1 - m2) / m1 * 100
    print(f"  {PLATFORM_SHORT[pt]:<12}: {pct:+5.1f}% (p={p:.4f})")

# Cross-platform variance at FP32
print("\n[3] Cross-platform spread at FP32 (Olist LightGBM):")
sub = df[(df['model']=='lightgbm') & (df['dataset']=='olist') & (df['precision']=='fp32')]
for _, row in sub.iterrows():
    print(f"  {PLATFORM_SHORT[row['platform']]:<12}: "
          f"{row['latency_mean']:6.1f} ± {row['latency_ci95']:4.1f} ms "
          f"(range: {row['latency_p5']:.1f}–{row['latency_p95']:.1f})")
spread = sub['latency_mean'].max() / sub['latency_mean'].min()
print(f"  Cross-platform spread ratio: {spread:.2f}x")

print("\nAll outputs saved.")
