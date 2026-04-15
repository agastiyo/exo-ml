#%%
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from src.utils.impute import EfficientPseudoGibbs
import src.utils.pca as PCA
import src.utils.datamatrix as DMatrix

#%%
# Set completeness threshold
tau = 0.7

# Load data matrices using datamatrix module
X = DMatrix.X(tau)
X_init = DMatrix.X_init(tau)
feature_names = DMatrix.feature_names(tau)

# Cut to complete case of X
X_known = X[np.isfinite(X).all(axis=1)]
print(f"Dataset shape: {X_known.shape}")
print(f"Number of features (tau={tau}): {len(feature_names)}")
print(f"Features: {feature_names}")

# Function to quantify the distance between two matrices
def RMSE_masked(X_original,X_imputed,mask):
  return np.sqrt(np.mean( (X_original[mask]-X_imputed[mask])**2 ))

# Function to quantify the distance between two eigenvalue spectra
def eig_log_l2(true, imp, eps=1e-8):
  return np.sqrt(np.sum((np.log(true + eps) - np.log(imp + eps))**2))

# MNAR mask a dataset with a given missing proportion
# Follows the Missing Based on Unobserved Variables (MBUV) strategy
def MNAR_mask(X_known, prop_missing):
  N, P = X_known.shape
  X_masked = X_known.copy()
  mask = np.zeros((N, P), dtype=bool)
  
  # 1. Generate unobserved feature from standard normal distribution
  f_unobserved = np.random.standard_normal(N)
  
  # 2. Rank by extremity: sort by absolute z-score values (already standardized)
  extremity_indices = np.argsort(-np.abs(f_unobserved))  # descending order
  
  # 3. Mask row features for top observations
  n_rows_to_mask = int(np.ceil(prop_missing * P))
  n_target_missing = int(prop_missing * N * P)
  
  # For each of the top rows ranked by extremity
  for row_idx in extremity_indices[:n_rows_to_mask]:
    # Randomly select one feature to keep (not mask)
    keep_feature = np.random.randint(0, P)
    # Mask all features except the one we keep
    for col_idx in range(P):
      if col_idx != keep_feature:
        X_masked[row_idx, col_idx] = np.nan
        mask[row_idx, col_idx] = True
  
  # 4. Count current masked cells
  current_masked = np.sum(mask)
  
  # Fill remaining mask if needed
  if current_masked < n_target_missing:
    remaining_to_mask = n_target_missing - current_masked
    # Find all unmasked cells
    unmasked_cells = np.where(~mask)
    # Randomly select remaining cells to mask
    if len(unmasked_cells[0]) > 0:
      random_indices = np.random.choice(len(unmasked_cells[0]), 
                                       size=min(remaining_to_mask, len(unmasked_cells[0])), 
                                       replace=False)
      for idx in random_indices:
        row = unmasked_cells[0][idx]
        col = unmasked_cells[1][idx]
        X_masked[row, col] = np.nan
        mask[row, col] = True
  
  return X_masked, mask

#%%
# Define the total runs
n_tot = 10
# Define the missing proportions you want to test
props_missing = np.arange(0.1, 0.7, 0.1)
# Define the number of runs for stochastic imputation methods
n_runs = 35

# Dictionary to track imputation performance
mean_diff_dict = {
  "Mean": [],
  "Median": [],
  "KNN": [],
  "MICE": [],
  "MissForest": [],
  "SWRF-Impute": []
}

# Dictionary to track eigenvalue spectra
eigval_dists_dict = {
  "Mean": [],
  "Median": [],
  "KNN": [],
  "MICE": [],
  "MissForest": [],
  "SWRF-Impute": []
}

# Store the true eigenvalue spectra
eigvals_true, _ = PCA.RunPCA(X_known)

#%%
for j in range(n_tot):
  print(f"Full Iteration {j+1}")
  diff_dict = {
    "Mean": [],
    "Median": [],
    "KNN": [],
    "MICE": [],
    "MissForest": [],
    "SWRF-Impute": []
  }
  
  eigvals_dict = {
    "Mean": [],
    "Median": [],
    "KNN": [],
    "MICE": [],
    "MissForest": [],
    "SWRF-Impute": []
  }
  
  # Loop through all missingness levels
  for prop_missing in props_missing:
    print(prop_missing)
    
    # Create the mask for this missingness level
    X_masked, mask = MNAR_mask(X_known, prop_missing)
    
    # --------------------------------------------

    # Mean
    X_mean = SimpleImputer(strategy='mean').fit_transform(X_masked)
    print("Mean done")

    # Median
    X_median = SimpleImputer(strategy='median').fit_transform(X_masked)
    print("Median done")

    # KNN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_masked)
    X_knn = KNNImputer(n_neighbors=5).fit_transform(X_scaled)
    X_knn = scaler.inverse_transform(X_knn)
    print("KNN done")

    # MissForest (RF IterativeImputer)
    rf = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=0)
    rf_imputer = IterativeImputer(estimator=rf, max_iter=10, random_state=0)
    X_mf = rf_imputer.fit_transform(X_masked)
    print("MissForest done")
    
    # SWRF-Impute
    X_out = EfficientPseudoGibbs(X_masked,X_init)
    X_stack = np.stack(X_out, axis=0)
    X_pgi = np.mean(X_stack, axis=0)
    print("SWRF-Impute Done")
    
    # MICE
    mice_runs = []
    mice_eigs = []
    
    for _ in range(n_runs):
      X_mice = IterativeImputer(max_iter=10,random_state=None,sample_posterior=True).fit_transform(X_masked)
      mice_runs.append(RMSE_masked(X_known, X_mice, mask))
      eig_mice, _ = PCA.RunPCA(X_mice)
      mice_eigs.append(eig_mice)
    print("MICE done")
    
    diff_dict["MICE"].append(np.mean(mice_runs))
    eigvals_dict["MICE"].append( eig_log_l2( eigvals_true, np.mean(mice_eigs, axis=0) ) )
    
    # --------------------------------------------
    
    # Save the results for this missingness level
    for name, X_imp in [("Mean", X_mean),("Median", X_median),("KNN", X_knn),
                        ("MissForest", X_mf),("SWRF-Impute", X_pgi)]:
      diff_dict[name].append(RMSE_masked(X_known, X_imp, mask))
      eigvals, _ = PCA.RunPCA(X_imp)
      eigvals_dict[name].append(eig_log_l2(eigvals_true,eigvals))

  # Save the results for this iteration through missingness levels
  for name in ["Mean", "Median", "KNN", "MissForest", "MICE", "SWRF-Impute"]:
    mean_diff_dict[name].append(diff_dict[name])
    eigval_dists_dict[name].append(eigvals_dict[name])

save_dir = "output/mnar_validation_runs"
os.makedirs(save_dir, exist_ok=True)

np.save(f"{save_dir}/mean_diff.npy", mean_diff_dict)
np.save(f"{save_dir}/eigval_dists.npy", eigval_dists_dict)
np.save(f"{save_dir}/props_missing.npy", props_missing)

plt.figure(figsize=(10, 6))

for name in ["Mean", "Median", "KNN", "MissForest", "MICE", "SWRF-Impute"]:
  means = np.mean(mean_diff_dict[name], axis=0)
  stds = np.std(mean_diff_dict[name], axis=0)
  plt.errorbar(props_missing, means, yerr=stds, marker='o', label=name)

plt.xlabel("Fraction of Missing Data", fontsize=12)
plt.ylabel("RMSE (on masked entries)", fontsize=12)
plt.title("MNAR Imputation Performance", fontsize=14)

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{save_dir}/performance.png")

plt.figure(figsize=(10, 6))

for name in ["Mean", "Median", "KNN", "MissForest", "MICE", "SWRF-Impute"]:
  means = np.mean(eigval_dists_dict[name], axis=0)
  stds = np.std(eigval_dists_dict[name], axis=0)
  plt.errorbar(props_missing, means, yerr=stds, marker='o', label=name)

plt.xlabel("Fraction of Missing Data", fontsize=12)
plt.ylabel("Log L2 dist from true Eigval spectra", fontsize=12)
plt.title("MNAR Pattern Destruction", fontsize=14)

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{save_dir}/pattern.png")
  
# %%
