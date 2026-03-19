#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from src.utils.impute import PseudoGibbsImputer

#%%
df = pd.read_csv('data/cleaned/STELLARHOSTS.csv',comment='#')
gaia_dir = "data/cleaned/gaia_arrays"
save_dir = "data/imputation_test"
feature_cols = ['sy_pnum', 'sy_snum', 'st_teff', 'st_rad', 'st_mass', 'st_met_FeH', 'st_lum', 'st_logg', 'st_age']

def RMSE_masked(X_original,X_imputed,mask):
  return np.sqrt(np.mean( (X_original[mask]-X_imputed[mask])**2 ))

# Create matrix X
X = df[feature_cols].to_numpy()

# Create matrix X_init
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

# Cut to complete case of X
X_known = X[np.isfinite(X).all(axis=1)]
print(X_known.shape)

# Artificially mask X_known
props_missing = np.arange(0.1, 1, 0.1)
n_runs = 5

meanDiff, medianDiff, KNNdiff, MICEdiff, MissForestDiff, PGIdiff = ([] for _ in range(6))

for prop_missing in props_missing:
  print(prop_missing)
  mean_runs, median_runs, knn_runs, mice_runs, mf_runs, pgi_runs = ([] for _ in range(6))
  
  for _ in range(n_runs):
    print(_)
    X_masked = X_known.copy()
    
    n_total = X_known.size
    n_missing = int(prop_missing * n_total)
    
    flat_idx = np.random.choice(n_total, n_missing, replace=False)
    row_idx, col_idx = np.unravel_index(flat_idx, X_known.shape)
    X_masked[row_idx, col_idx] = np.nan
    
    mask = np.isnan(X_masked)

    # Mean
    X_mean = SimpleImputer(strategy='mean').fit_transform(X_masked)
    mean_runs.append(RMSE_masked(X_known, X_mean, mask))

    # Median
    X_median = SimpleImputer(strategy='median').fit_transform(X_masked)
    median_runs.append(RMSE_masked(X_known, X_median, mask))

    # KNN (scaled)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_masked)
    X_knn = KNNImputer(n_neighbors=5).fit_transform(X_scaled)
    X_knn = scaler.inverse_transform(X_knn)
    knn_runs.append(RMSE_masked(X_known, X_knn, mask))

    # MICE
    X_mice = IterativeImputer(max_iter=10, random_state=0, sample_posterior=True).fit_transform(X_masked)
    mice_runs.append(RMSE_masked(X_known, X_mice, mask))

    # MissForest
    rf = RandomForestRegressor(n_estimators=50, random_state=0)
    rf_imputer = IterativeImputer(
      estimator=rf,
      max_iter=10,
      random_state=0
    )
    X_mf = rf_imputer.fit_transform(X_masked)
    mf_runs.append(RMSE_masked(X_known, X_mf, mask))

    # PGI
    regressor = RandomForestRegressor(n_estimators=50, n_jobs=-1)
    X_pgi, _ = PseudoGibbsImputer(X_masked, X_init, regressor, save_directory=None, tot_iters=100)
    pgi_runs.append(RMSE_masked(X_known, X_pgi, mask))

  meanDiff.append(np.mean(mean_runs))
  medianDiff.append(np.mean(median_runs))
  KNNdiff.append(np.mean(knn_runs))
  MICEdiff.append(np.mean(mice_runs))
  MissForestDiff.append(np.mean(mf_runs))
  PGIdiff.append(np.mean(pgi_runs))

#%%
plt.figure(figsize=(10, 6))

plt.plot(props_missing, meanDiff, marker='o', label='Mean')
plt.plot(props_missing, medianDiff, marker='o', label='Median')
plt.plot(props_missing, KNNdiff, marker='o', label='KNN')
plt.plot(props_missing, MICEdiff, marker='o', label='MICE')
plt.plot(props_missing, MissForestDiff, marker='o', label='MissForest')
plt.plot(props_missing, PGIdiff, marker='o', linewidth=3, label='GSimp-RF+')

plt.xlabel("Fraction of Missing Data", fontsize=12)
plt.ylabel("RMSE (on masked entries)", fontsize=12)
plt.title("Imputation Performance vs Missingness", fontsize=14)

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
# %%
