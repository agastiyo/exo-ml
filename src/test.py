#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils.impute import EfficientPseudoGibbs

#%%
df = pd.read_csv('data/cleaned/STELLARHOSTS.csv',comment='#') # Stellarhost dataframe
gaia_dir = "data/cleaned/gaia_arrays"
save_dir = "data/imputed"
feature_cols = ['sy_pnum', 'sy_snum', 'st_teff', 'st_rad', 'st_mass', 'st_met_FeH', 'st_lum', 'st_logg', 'st_age']
# This is the order the features will be organized in the data matrix X, so make sure that the initializer matrix has the features in the same order

# Build data matrix X
X = df[feature_cols].to_numpy()

# Build the initializer matrix X_init
# Padding mismatched lengths with NaNs
rows = [
  np.ones(1),
  np.ones(1),
  np.load(f"{gaia_dir}/teff.npy"),
  np.load(f"{gaia_dir}/radius.npy"),
  np.load(f"{gaia_dir}/mass.npy"),
  np.load(f"{gaia_dir}/feh.npy"),
  np.load(f"{gaia_dir}/lum.npy"),
  np.load(f"{gaia_dir}/logg.npy"),
  np.load(f"{gaia_dir}/age.npy")
]

max_len = max(r.size for r in rows)

X_init = np.full((len(rows), max_len), np.nan, dtype=float)

for i, r in enumerate(rows):
  X_init[i, :r.size] = r

X_init = X_init.T

X_output = EfficientPseudoGibbs(X,X_init,n_iters=30,n_trees=50,burn_in=15,stochastic_str=1)

#%%
X_stack = np.stack(X_output, axis=0)
X_mean_imputed = np.mean(X_stack, axis=0)

n_features = X.shape[1]

for i in range(n_features):
  plt.figure()
  
  # Original data histogram
  plt.hist(X[:, i], bins=50, alpha=0.5, label='Original', density=True)
  
  # Imputed mean histogram
  plt.hist(X_mean_imputed[:, i], bins=50, alpha=0.5, label='Imputed (Mean of samples)', density=True)
  
  plt.title(f"Feature: {feature_cols[i]}")
  plt.xlabel("Value")
  plt.ylabel("Density")
  plt.legend()
  
  plt.show()
# %%
