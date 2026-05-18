"""Post-processing modules used downstream of the LightGBM pipeline.

Includes:
- ``kalman.NWPBiasKalman`` — local-level smoother for NWP↔ERA5 wind-speed
  bias. Outputs a smoothed bias column and a bias-corrected NWP wind column
  for consumption as FEATURES by the booster (not as a post-hoc adjustment
  on predictions — the previous static bias correction hurt Fold-5, see
  PROJECT.md §10 item 11).
- ``calibration.PerMonthAffine`` — an opt-in per-month affine fit on OOF
  predictions from folds 1–4, applied to Fold-5 or test predictions. Only
  shipped when the Fold-5 lift is ≥ 0.02 pp (TODO.md stop condition).
"""
