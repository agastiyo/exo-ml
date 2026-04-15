#%%
import numpy as np
import warnings
import src.utils.datamatrix as DMatrix
from src.validation._common import run_validation

warnings.filterwarnings("ignore")

#%%
tau = 0.7
X             = DMatrix.X(tau)
X_init        = DMatrix.X_init(tau)
feature_names = DMatrix.feature_names(tau)

X_known = X[np.isfinite(X).all(axis=1)]
print(f"Complete-case shape: {X_known.shape}")
print(f"Features (tau={tau}): {feature_names}")

#%%
def MCAR_mask(X_known, prop_missing, rng):
  X_masked = X_known.copy()
  n_total  = X_known.size
  n_missing = int(prop_missing * n_total)
  flat_idx  = rng.choice(n_total, n_missing, replace=False)
  row_idx, col_idx = np.unravel_index(flat_idx, X_known.shape)
  X_masked[row_idx, col_idx] = np.nan
  return X_masked, np.isnan(X_masked)

#%%
run_validation(MCAR_mask, "mcar", X_known, X_init, n_tot=1, n_runs=5)

# %%
