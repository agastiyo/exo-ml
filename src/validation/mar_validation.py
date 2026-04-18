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

# Use synthetic stellar host population (fully observed, N=15000, P=9)
X_known, feature_names = DMatrix.synthetic_stellarhosts(n_stars=15000)
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
  N, P = X_known.shape
  n_cells_target = int(np.floor(prop_missing * N * P))

  # Randomly select a trigger feature
  trigger_idx = rng.integers(0, P)

  # Rank observations by extremity of the trigger feature
  trigger = X_known[:, trigger_idx]
  z_scores = (trigger - np.nanmean(trigger)) / (np.nanstd(trigger) + 1e-10)
  extremity_order = np.argsort(-np.abs(z_scores))  # descending

  mask = np.zeros((N, P), dtype=bool)

  # Mask all non-trigger columns in the most extreme rows.
  n_rows_to_mask = int(np.ceil(prop_missing * N))
  masked_count = 0

  for i in range(n_rows_to_mask):
    row_idx = extremity_order[i]
    for col_idx in range(P):
      if col_idx != trigger_idx:
        mask[row_idx, col_idx] = True
        masked_count += 1

  # Top up to the target cell count with uniform random cells if needed
  if masked_count < n_cells_target:
    remaining = n_cells_target - masked_count
    unmasked  = np.where(~mask)
    n_avail   = len(unmasked[0])
    if n_avail > 0:
      chosen = rng.choice(n_avail, size=min(remaining, n_avail), replace=False)
      mask[unmasked[0][chosen], unmasked[1][chosen]] = True

  X_masked = X_known.copy()
  X_masked[mask] = np.nan
  return X_masked, mask

#%%
run_validation(MAR_mask, "mar", X_known, bounds_list, n_tot=1, n_runs=5)

# %%
