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
def MNAR_mask(X_known, prop_missing, rng):
  """
  Missing Based on Unobserved Variables (MBUV).

  An unobserved feature drawn from a standard normal determines which rows
  are most extreme. In those rows all-but-one column is masked, creating
  missingness that depends on an unseen latent variable (true MNAR).
  """
  N, P = X_known.shape
  n_target = int(prop_missing * N * P)

  # Unobserved latent variable
  f_unobserved  = rng.standard_normal(N)
  extremity_idx = np.argsort(-np.abs(f_unobserved))  # most extreme first

  mask = np.zeros((N, P), dtype=bool)

  # Mask all columns except one random survivor in each extreme row.
  # Scale the number of affected rows with N so coverage is consistent
  # across different missingness levels.
  n_rows_to_mask = int(np.ceil(prop_missing * N))

  for row_idx in extremity_idx[:n_rows_to_mask]:
    keep_col = rng.integers(0, P)
    for col_idx in range(P):
      if col_idx != keep_col:
        mask[row_idx, col_idx] = True

  # Top up to the exact target without random fill-in that would dilute
  # the MNAR structure. Instead extend to additional extreme rows.
  current = np.sum(mask)
  if current < n_target:
    remaining = n_target - current
    unmasked  = np.where(~mask)
    n_avail   = len(unmasked[0])
    if n_avail > 0:
      # Prefer cells in already-affected rows before touching new rows
      chosen = rng.choice(n_avail, size=min(remaining, n_avail), replace=False)
      mask[unmasked[0][chosen], unmasked[1][chosen]] = True

  X_masked = X_known.copy()
  X_masked[mask] = np.nan
  return X_masked, mask

#%%
run_validation(MNAR_mask, "mnar", X_known, X_init)

# %%
