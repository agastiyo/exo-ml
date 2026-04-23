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
def MNAR_mask(X_known, prop_missing, rng):
  """
  Stochastic tail MNAR: each cell is assigned a missing probability based on
  its z-score magnitude in its feature. Cells are then sampled stochastically
  to reach prop_missing * N * P total missing cells. Missingness depends on
  the feature's own value — physically motivated for astrophysical data where
  extreme measurements are harder to record.
  """
  N, P = X_known.shape
  mask = np.zeros((N, P), dtype=bool)

  # Compute |z-scores| per feature
  z_abs = np.zeros((N, P))
  for j in range(P):
    col = X_known[:, j]
    z = (col - np.nanmean(col)) / (np.nanstd(col) + 1e-10)
    z_abs[:, j] = np.abs(z)

  # Assign missing probabilities proportional to |z|^2 (quadratic to emphasize extremes)
  probs = z_abs ** 2
  probs = probs / np.sum(probs)  # normalize to probability distribution

  # Sample which cells to mask, targeting prop_missing fraction of all cells
  total_cells = N * P
  n_to_mask = np.floor(prop_missing * N * P)

  # Draw cells without replacement according to z-score probabilities
  flat_indices = rng.choice(
    total_cells,
    size=n_to_mask,
    replace=False,
    p=probs.flatten()
  )

  # Convert flat indices back to 2D and set mask
  rows, cols = np.unravel_index(flat_indices, (N, P))
  mask[rows, cols] = True

  X_masked = X_known.copy()
  X_masked[mask] = np.nan
  return X_masked, mask

#%%
run_validation(MNAR_mask, "mnar", X_known, bounds_list, n_tot=1, n_runs=35)

# %%
