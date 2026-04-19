#%%
import numpy as np
import src.utils.datamatrix as DMatrix
from src.validation._common import run_validation
import warnings

warnings.filterwarnings("ignore")

#%%
# tau = 0.7
# X             = DMatrix.X(tau)
# feature_names = DMatrix.feature_names(tau)
# X_known = X[np.isfinite(X).all(axis=1)]  # Complete-case from real STELLARHOSTS data
# print(f"Complete-case shape: {X_known.shape}")

# Use synthetic stellar host population (fully observed, N=500, P=9)
X_known, feature_names = DMatrix.synthetic_stellarhosts(n_stars=500)
print(f"Synthetic population shape: {X_known.shape}")
print(f"Features: {feature_names}")

# %%
# Order: [sy_pnum, sy_snum, st_teff, st_rad, st_mass, st_met_FeH, st_lum, st_logg, st_age]
bounds_list = [
  (1, None),  # sy_pnum
  (1, None),  # sy_snum
  (0, None),  # st_teff  [K]
  (0, None),  # st_rad   [R_sun]
  (0, None),  # st_mass  [M_sun]
  (None, None),  # st_met_FeH [dex]
  (0, None),  # st_lum   [W]
  (0, None),  # st_logg  [log10 cm/s^2]
  (0, 13.8),  # st_age   [Gyr]
]

#%%
def MAR_mask(X_known, prop_missing, rng):
  """
  Extremity-weighted MAR: a random observed feature is the trigger. Non-trigger
  cells are sampled without replacement with row probability proportional to the
  trigger feature's |z-score|, so more extreme rows are more likely to be masked
  but less extreme rows are not excluded. Exact prop_missing is preserved.
  """
  N, P = X_known.shape
  n_cells_target = int(np.floor(prop_missing * N * P))

  trigger_idx = rng.integers(0, P)
  trigger = X_known[:, trigger_idx]
  z_scores = (trigger - np.nanmean(trigger)) / (np.nanstd(trigger) + 1e-10)
  row_weights = np.abs(z_scores)
  row_weights = row_weights / row_weights.sum()

  # Build flat index of all eligible (non-trigger) cells with per-cell weights
  eligible_rows, eligible_cols, cell_weights = [], [], []
  for col in range(P):
    if col == trigger_idx:
      continue
    eligible_rows.extend(range(N))
    eligible_cols.extend([col] * N)
    cell_weights.extend(row_weights.tolist())

  eligible_rows = np.array(eligible_rows)
  eligible_cols = np.array(eligible_cols)
  cell_weights  = np.array(cell_weights)
  cell_weights /= cell_weights.sum()

  n_to_mask = min(n_cells_target, len(eligible_rows))
  chosen = rng.choice(len(eligible_rows), size=n_to_mask, replace=False, p=cell_weights)

  mask = np.zeros((N, P), dtype=bool)
  mask[eligible_rows[chosen], eligible_cols[chosen]] = True

  X_masked = X_known.copy()
  X_masked[mask] = np.nan
  return X_masked, mask

#%%
run_validation(MAR_mask, "mar", X_known, bounds_list, n_tot=1, n_runs=35)

# %%
