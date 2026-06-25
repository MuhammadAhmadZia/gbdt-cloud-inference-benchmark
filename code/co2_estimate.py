# ==============================================================================
# CARBON SAVING ESTIMATE
#
# Illustrative estimate of the annual carbon saving from a compiled-runtime
# latency reduction, computed from the measured per-inference energy. All
# assumptions (traffic, energy saving fraction, grid intensity) are stated
# explicitly so the figure is fully reproducible and can be scaled.
#
# Energy basis: measured per-10k-row-batch energy for LightGBM at FP64,
# taken from the benchmark results. Adjust PREDICTIONS_PER_DAY and the basis
# to match the scenario being reported.
# ==============================================================================

PREDICTIONS_PER_DAY    = 1_000_000_000   # illustrative service traffic
GRID_INTENSITY_G_KWH   = 480             # world-average grid carbon intensity
ENERGY_SAVING_FRACTION = 0.50            # compiled-runtime latency/energy saving
DAYS_PER_YEAR          = 365

# Measured energy per 10,000-row batch (kWh), LightGBM FP64:
MEASURED = {
    ('colab', 'olist'):    2.161e-06,
    ('colab', 'nyc_taxi'): 8.172e-06,
}
BATCH_ROWS = 10_000

def estimate(platform, dataset):
    batch_kwh = MEASURED[(platform, dataset)]
    per_pred_kwh = batch_kwh / BATCH_ROWS
    saved_per_pred = per_pred_kwh * ENERGY_SAVING_FRACTION
    annual_kwh = saved_per_pred * PREDICTIONS_PER_DAY * DAYS_PER_YEAR
    annual_co2_kg = annual_kwh * GRID_INTENSITY_G_KWH / 1000.0
    return per_pred_kwh, annual_kwh, annual_co2_kg

print("Assumptions:")
print(f"  predictions/day = {PREDICTIONS_PER_DAY:,}")
print(f"  energy saving    = {ENERGY_SAVING_FRACTION:.0%}")
print(f"  grid intensity   = {GRID_INTENSITY_G_KWH} g CO2/kWh")
print(f"  basis            = LightGBM FP64, per-10k-row batch\n")
print(f"{'platform':8} {'dataset':9} {'per-pred kWh':>14} {'kWh saved/yr':>14} {'kg CO2/yr':>12}")
for (plat, ds) in MEASURED:
    pp, ann_kwh, ann_co2 = estimate(plat, ds)
    print(f"{plat:8} {ds:9} {pp:14.3e} {ann_kwh:14.2f} {ann_co2:12.2f}")
