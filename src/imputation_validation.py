#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from src.utils.impute import PseudoGibbsImputer
from src.utils.impute import EfficientPseudoGibbs
import src.utils.pca as PCA

#%%
# Loading all of the data
df = pd.read_csv('data/cleaned/STELLARHOSTS.csv',comment='#')
gaia_dir = "data/cleaned/gaia_arrays"
save_dir = "data/imputation_test"
feature_cols = ['sy_pnum', 'sy_snum', 'st_teff', 'st_rad', 'st_mass', 'st_met_FeH', 'st_lum', 'st_logg', 'st_age']

# Function to quantify the distance between two matrices
def RMSE_masked(X_original,X_imputed,mask):
  return np.sqrt(np.mean( (X_original[mask]-X_imputed[mask])**2 ))

# Function to calculate the Marchenko-Pastur bounds
def marchenko_pastur_bounds(X):
  """
  Computes MP bounds for covariance eigenvalues.
  Assumes X is standardized.
  """
  n, p = X.shape
  q = p / n
  
  sigma2 = 1.0  # since we standardize
  
  lambda_minus = sigma2 * (1 - np.sqrt(q))**2
  lambda_plus  = sigma2 * (1 + np.sqrt(q))**2
  
  return lambda_minus, lambda_plus

# Function to quantify the distance between two eigenvalue spectra
def eig_log_l2(true, imp, eps=1e-8):
  return np.sqrt(np.sum((np.log(true + eps) - np.log(imp + eps))**2))

#%%

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

#%%
# Define the total runs
n_tot = 10
# Define the missing proportions you want to test
props_missing = np.arange(0.1, 1, 0.1)
# Define the number of runs for stochastic imputation methods
n_runs = 5

# Dictionary to track imputation performance
mean_diff_dict = {
  "Mean": [],
  "Median": [],
  "KNN": [],
  "MICE": [],
  "MissForest": [],
  "GSimp-RF+": []
}

# Dictionary to track eigenvalue spectra
eigval_dists_dict = {
  "Mean": [],
  "Median": [],
  "KNN": [],
  "MICE": [],
  "MissForest": [],
  "GSimp-RF+": []
}

# Store the true eigenvalue spectra
eigvals_true, _ = PCA.RunPCA(X_known)
plt.plot(eigvals_true)

#%%
for j in range(n_tot):
  print(f"Full Iteration {j+1}")
  diff_dict = {
    "Mean": [],
    "Median": [],
    "KNN": [],
    "MICE": [],
    "MissForest": [],
    "GSimp-RF+": []
  }
  
  eigvals_dict = {
    "Mean": [],
    "Median": [],
    "KNN": [],
    "MICE": [],
    "MissForest": [],
    "GSimp-RF+": []
  }
  
  for prop_missing in props_missing:
    print(prop_missing)
    
    # ---- SINGLE MASK PER MISSINGNESS (important for fair comparison) ----
    X_masked = X_known.copy()
    
    n_total = X_known.size
    n_missing = int(prop_missing * n_total)
    
    flat_idx = np.random.choice(n_total, n_missing, replace=False)
    row_idx, col_idx = np.unravel_index(flat_idx, X_known.shape)
    X_masked[row_idx, col_idx] = np.nan
    
    mask = np.isnan(X_masked)

    # ------------------ DETERMINISTIC METHODS ------------------

    # Mean
    X_mean = SimpleImputer(strategy='mean').fit_transform(X_masked)
    diff_dict["Mean"].append(RMSE_masked(X_known, X_mean, mask))

    # Median
    X_median = SimpleImputer(strategy='median').fit_transform(X_masked)
    diff_dict["Median"].append(RMSE_masked(X_known, X_median, mask))

    # KNN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_masked)
    X_knn = KNNImputer(n_neighbors=5).fit_transform(X_scaled)
    X_knn = scaler.inverse_transform(X_knn)
    diff_dict["KNN"].append(RMSE_masked(X_known, X_knn, mask))

    # MissForest (RF IterativeImputer)
    rf = RandomForestRegressor(n_estimators=50, random_state=0)
    rf_imputer = IterativeImputer(estimator=rf, max_iter=10, random_state=0)
    X_mf = rf_imputer.fit_transform(X_masked)
    diff_dict["MissForest"].append(RMSE_masked(X_known, X_mf, mask))
    
    # ---- PCA for deterministic methods ----
    for name, X_imp in [("Mean", X_mean),("Median", X_median),("KNN", X_knn),("MissForest", X_mf)]:
      eigvals, _ = PCA.RunPCA(X_imp)
      eigvals_dict[name].append(eig_log_l2(eigvals_true,eigvals))

    # ------------------ STOCHASTIC METHODS ------------------

    mice_runs = []
    pgi_runs = []
    
    mice_eigs = []
    pgi_eigs = []
    
    # GSimp-RF+
    X_out = EfficientPseudoGibbs(X_masked,X_init)
    X_stack = np.stack(X_out, axis=0)
    X_pgi = np.mean(X_stack, axis=0)
    diff_dict["GSimp-RF+"].append(RMSE_masked(X_known, X_pgi, mask))
    eig_pgi, _  = PCA.RunPCA(X_pgi)
    eigvals_dict["GSimp-RF+"].append( eig_log_l2(eigvals_true,eig_pgi) )

    for _ in range(n_runs):
      print("  run:", _+1)

      # MICE
      X_mice = IterativeImputer(max_iter=10,random_state=None,sample_posterior=True).fit_transform(X_masked)

      mice_runs.append(RMSE_masked(X_known, X_mice, mask))

      # ---- PCA ----

      eig_mice, _ = PCA.RunPCA(X_mice)

      mice_eigs.append(eig_mice)

    diff_dict["MICE"].append(np.mean(mice_runs))
    eigvals_dict["MICE"].append( eig_log_l2( eigvals_true, np.mean(mice_eigs, axis=0) ) )
  
  for name in ["Mean", "Median", "KNN", "MissForest", "MICE", "GSimp-RF+"]:
    mean_diff_dict[name].append(diff_dict[name])
    eigval_dists_dict[name].append(eigvals_dict[name])
  
for name in mean_diff_dict:
  mean_diff_dict[name] = np.mean(mean_diff_dict[name], axis=0)

for name in eigval_dists_dict:
  eigval_dists_dict[name] = np.mean(eigval_dists_dict[name], axis=0)

#%%
os.makedirs("data/efficient_imputation_validation_dicts", exist_ok=True)

np.save(f"data/efficient_imputation_validation_dicts/mean_diff.npy", mean_diff_dict)
np.save(f"data/efficient_imputation_validation_dicts/eigval_dists.npy", eigval_dists_dict)
np.save(f"data/efficient_imputation_validation_dicts/props_missing.npy", props_missing)
#%%
plt.figure(figsize=(10, 6))

for name in ["Mean", "Median", "KNN", "MissForest", "MICE", "GSimp-RF+"]:
  plt.plot(props_missing, mean_diff_dict[name], marker='o', label=name)

plt.xlabel("Fraction of Missing Data", fontsize=12)
plt.ylabel("RMSE (on masked entries)", fontsize=12)
plt.title("Imputation Performance vs Missingness", fontsize=14)

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(10, 6))

for name in ["Mean", "Median", "KNN", "MissForest", "MICE", "GSimp-RF+"]:
  plt.plot(props_missing, eigval_dists_dict[name], marker='o', label=name)

plt.xlabel("Fraction of Missing Data", fontsize=12)
plt.ylabel("Log L2 dist from true Eigval spectra", fontsize=12)
plt.title("Pattern Destruction vs Missingness", fontsize=14)

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
  
# %%
