# ==============================================================================
# HIERARCHICAL LATENCY ANALYSIS
#
# Statistical analysis of the repeated latency measurements. Because the timed
# repetitions within a configuration are not independent, this fits a linear
# mixed-effects model on log-latency with a configuration-level random
# intercept, reports the intraclass correlation, and gives an interaction-aware
# variance decomposition over engine, precision, and dataset.
#
# Reads the detailed benchmark output (results_detailed_*.json), which retains
# the raw per-repetition latencies. No inference benchmarking is performed.
#
# INPUT FILES:
#   {BASE_DIR}/results_detailed_*.json
#
# INSTALL:
#   pip install pandas numpy statsmodels -q
# ==============================================================================

import glob, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

BASE_DIR = '/content/drive/MyDrive/paper2_benchmark'
JSON_GLOB = f'{BASE_DIR}/results_detailed_*.json'

files = sorted(glob.glob(JSON_GLOB))
print("Found:", files)

# ---- 1. load + explode raw repetitions to long format ----
long_rows = []
for path in files:
    d = json.load(open(path))
    plat = d['hardware']['platform_tag']
    for e in d['experiments']:
        raw = e['latency_ms'].get('raw')
        if not raw:
            continue
        for rep_i, ms in enumerate(raw):
            long_rows.append({'platform': plat, 'dataset': e['dataset'],
                              'engine': e['model'], 'precision': e['precision'],
                              'rep': rep_i, 'latency_ms': ms, 'log_latency': np.log(ms)})

long = pd.DataFrame(long_rows)
if long.empty:
    raise SystemExit("No raw latencies found in results_detailed_*.json.")

long['config'] = (long['platform'] + '|' + long['dataset'] + '|' +
                  long['engine'] + '|' + long['precision'])
print(f"\nLoaded {len(long)} repetitions across {long['config'].nunique()} "
      f"configurations, {long['platform'].nunique()} platform(s).\n")

# ---- 2. mixed-effects model: random intercept per configuration ----
fixed = "log_latency ~ C(engine) * C(precision) + C(dataset)"
if long['platform'].nunique() > 1:
    fixed += " + C(platform)"

print("=" * 70, "\nMIXED-EFFECTS MODEL (random intercept per configuration)\n", "=" * 70, sep="")
mfit = smf.mixedlm(fixed, long, groups=long['config']).fit(method='lbfgs', reml=True)
print(mfit.summary())

grp_var, resid_var = float(mfit.cov_re.iloc[0, 0]), float(mfit.scale)
icc = grp_var / (grp_var + resid_var)
print(f"\nBetween-config variance: {grp_var:.5f}")
print(f"Within-config residual : {resid_var:.5f}")
print(f"Intraclass correlation (ICC): {icc:.3f}")

# ---- 3. interaction-aware variance decomposition ----
print("\n", "=" * 70, "\nVARIANCE DECOMPOSITION (Type-II ANOVA on raw repetitions)\n", "=" * 70, sep="")
ols_formula = "log_latency ~ C(engine) * C(precision) * C(dataset)"
if long['platform'].nunique() > 1:
    ols_formula += " * C(platform)"
ols = smf.ols(ols_formula, data=long).fit()
aov = sm.stats.anova_lm(ols, typ=2)
aov['eta_sq'] = aov['sum_sq'] / aov['sum_sq'].sum()
aov = aov.sort_values('eta_sq', ascending=False)
print(aov[['sum_sq', 'df', 'F', 'PR(>F)', 'eta_sq']].to_string())

aov.to_csv(f'{BASE_DIR}/hierarchical_anova.csv')
print(f"\nSaved {BASE_DIR}/hierarchical_anova.csv")
