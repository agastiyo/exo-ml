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
def RSHM_mask(X_known, prop_missing, rng):
  raise NotImplementedError()
  return X_masked, mask

#%%
run_validation(RSHM_mask, "rshm", X_known, bounds_list, n_tot=1, n_runs=35)

# %%
