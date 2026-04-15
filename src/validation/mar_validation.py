#%%
import numpy as np
import src.utils.datamatrix as DMatrix
from src.validation._common import run_validation

#%%
tau = 0.7
X             = DMatrix.X(tau)
X_init        = DMatrix.X_init(tau)
feature_names = DMatrix.feature_names(tau)

X_known = X[np.isfinite(X).all(axis=1)]
print(f"Complete-case shape: {X_known.shape}")
print(f"Features (tau={tau}): {feature_names}")

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
run_validation(MAR_mask, "mar", X_known, X_init)

# %%
