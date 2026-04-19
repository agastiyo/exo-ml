import numpy as np
import os
from datetime import datetime
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from src.utils.impute import SWRF_Impute
import src.utils.pca as PCA

METHOD_NAMES       = ["Mean", "Median", "KNN", "MICE", "MissForest", "SWRF-Impute"]
STOCHASTIC_METHODS = ["MICE", "SWRF-Impute", "MissForest"]   # true posterior samplers — IQR coverage computed
N_RUNS_MF          = 5                          # MissForest: near-deterministic; small ensemble for stable mean only

def RMSE_masked(X_original, X_imputed, mask):
  scaler = StandardScaler()
  X_og = scaler.fit_transform(X_original)
  X_imp = scaler.fit_transform(X_imputed)
  return np.sqrt(np.mean((X_og[mask] - X_imp[mask])**2))

def eig_log_l2(true, imp, eps=1e-8):
  return np.sqrt(np.sum((np.log(true + eps) - np.log(imp + eps))**2))

def iqr_coverage(X_true, ensemble, mask):
  """
  Fraction of masked true values that fall inside the cell-wise IQR
  (25th to 75th percentile) of an ensemble of imputed matrices.

  A well-calibrated imputer gives coverage ≈ 0.5. Coverage well below 0.5
  indicates under-dispersed (over-confident) draws; coverage well above
  0.5 indicates over-dispersed (under-confident) draws.

  Parameters
  ----------
  X_true   : (N, P) ndarray    ground-truth complete-case matrix
  ensemble : (K, N, P) ndarray K imputed matrices stacked along axis 0
  mask     : (N, P) bool       True where the cell was held out
  """
  q25 = np.percentile(ensemble, 25, axis=0)
  q75 = np.percentile(ensemble, 75, axis=0)
  within = (X_true[mask] >= q25[mask]) & (X_true[mask] <= q75[mask])
  return float(np.mean(within))

def run_validation(mask_fn, mechanism_name, X_known, bounds_list,
                   n_tot=10, props_missing=None, n_runs=35, seed=None):
  """
  Run the full imputation validation harness for a given missingness mechanism.

  Saves mean_diff.npy, eigval_dists.npy, and props_missing.npy into a
  timestamped subfolder under output/<mechanism_name>_validation_runs/.

  Parameters
  ----------
  mask_fn         : callable(X_known, prop_missing, rng) -> (X_masked, mask)
  mechanism_name  : str   e.g. "mcar", "mar", "mnar"
  X_known         : ndarray  complete-case data matrix
  bounds_list     : list of P tuples (lo, hi) — physical bounds per feature for
                    SWRF-Impute; either element may be None (unbounded on that side)
  n_tot           : int   number of outer repetitions
  props_missing   : array-like  missingness proportions to sweep
  n_runs          : int   draws per missingness level for MICE and SWRF-Impute
                          (true posterior samplers). MissForest uses N_RUNS_MF
                          regardless — it is near-deterministic and only needs a
                          small ensemble to stabilise the pooled mean.
  seed            : int   base random seed

  Returns
  -------
  mean_diff_dict, eigval_dists_dict, coverage_dict, props_missing, save_dir
  """
  if props_missing is None:
    props_missing = np.arange(0.1, 0.7, 0.1)
  props_missing = np.asarray(props_missing)
  
  rng = np.random.default_rng()
  if seed:
    rng = np.random.default_rng(seed)
  eigvals_true, _ = PCA.RunPCA(X_known)

  mean_diff_dict    = {name: [] for name in METHOD_NAMES}
  eigval_dists_dict = {name: [] for name in METHOD_NAMES}
  coverage_dict     = {name: [] for name in STOCHASTIC_METHODS}

  for j in range(n_tot):
    print(f"Iteration {j+1}/{n_tot}")
    diff_dict    = {name: [] for name in METHOD_NAMES}
    eigvals_dict = {name: [] for name in METHOD_NAMES}
    cov_dict     = {name: [] for name in STOCHASTIC_METHODS}

    for prop_missing in props_missing:
      print(f"  prop_missing={prop_missing:.2f}")

      # Each (iteration, prop_missing) pair gets a child rng derived from the
      # master rng so runs are reproducible but independent of each other.
      iter_rng = np.random.default_rng(rng.integers(0, 2**31))
      X_masked, mask = mask_fn(X_known, prop_missing, iter_rng)

      # ------------------------------------------------------------------
      # Deterministic methods — run once

      X_mean   = SimpleImputer(strategy='mean').fit_transform(X_masked)
      print(f"    Mean Done")
      X_median = SimpleImputer(strategy='median').fit_transform(X_masked)
      print(f"    Median Done")

      scaler  = StandardScaler()
      X_knn   = KNNImputer(n_neighbors=5).fit_transform(scaler.fit_transform(X_masked))
      X_knn   = scaler.inverse_transform(X_knn)
      print(f"    KNN Done")

      for name, X_imp in [("Mean", X_mean), ("Median", X_median), ("KNN", X_knn)]:
        diff_dict[name].append(RMSE_masked(X_known, X_imp, mask))
        eig, _ = PCA.RunPCA(X_imp)
        eigvals_dict[name].append(eig_log_l2(eigvals_true, eig))

      # ------------------------------------------------------------------
      # SWRF-Impute — single Gibbs chain; draws are averaged into a pooled matrix.
      # A child rng is derived from the master so the chain is fully reproducible.

      swrf_rng = np.random.default_rng(rng.integers(0, 2**31))
      X_out    = SWRF_Impute(X_masked, bounds_list, rng=swrf_rng)
      X_pgi    = np.mean(X_out, axis=0)
      print(f"    SWRF Done")

      diff_dict["SWRF-Impute"].append(RMSE_masked(X_known, X_pgi, mask))
      eig, _ = PCA.RunPCA(X_pgi)
      eigvals_dict["SWRF-Impute"].append(eig_log_l2(eigvals_true, eig))
      cov_dict["SWRF-Impute"].append(iqr_coverage(X_known, X_out, mask))

      # ------------------------------------------------------------------
      # MissForest — near-deterministic point estimator. Run N_RUNS_MF times
      # with different seeds only to stabilise the pooled mean; bootstrap
      # variance is too small for IQR coverage to be meaningful.

      mf_matrices = []
      for k in range(N_RUNS_MF):
        rf_state = int(rng.integers(0, 2**31))
        rf  = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=rf_state)
        X_mf = IterativeImputer(estimator=rf, max_iter=10,
                                random_state=rf_state).fit_transform(X_masked)
        mf_matrices.append(X_mf)
        print(f"    MissForest {k+1}/{N_RUNS_MF}")

      mf_ensemble = np.stack(mf_matrices, axis=0)
      X_mf_pooled = np.mean(mf_ensemble, axis=0)
      diff_dict["MissForest"].append(RMSE_masked(X_known, X_mf_pooled, mask))
      eig, _ = PCA.RunPCA(X_mf_pooled)
      eigvals_dict["MissForest"].append(eig_log_l2(eigvals_true, eig))
      cov_dict["MissForest"].append(iqr_coverage(X_known, mf_ensemble, mask))

      # ------------------------------------------------------------------
      # MICE — true posterior sampler. Run n_runs times; pool cell-wise for
      # RMSE and eigval; full ensemble for IQR coverage.

      mice_matrices = []
      for k in range(n_runs):
        mice_state = int(rng.integers(0, 2**31))
        X_mice = IterativeImputer(max_iter=10, random_state=mice_state,
                                  sample_posterior=True).fit_transform(X_masked)
        mice_matrices.append(X_mice)
        print(f"    MICE {k+1}/{n_runs}")

      mice_ensemble = np.stack(mice_matrices, axis=0)
      X_mice_pooled = np.mean(mice_ensemble, axis=0)
      diff_dict["MICE"].append(RMSE_masked(X_known, X_mice_pooled, mask))
      eig, _ = PCA.RunPCA(X_mice_pooled)
      eigvals_dict["MICE"].append(eig_log_l2(eigvals_true, eig))
      cov_dict["MICE"].append(iqr_coverage(X_known, mice_ensemble, mask))

    for name in METHOD_NAMES:
      mean_diff_dict[name].append(diff_dict[name])
      eigval_dists_dict[name].append(eigvals_dict[name])
    for name in STOCHASTIC_METHODS:
      coverage_dict[name].append(cov_dict[name])

  # ------------------------------------------------------------------
  # Bundle outputs into a timestamped folder

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  save_dir  = os.path.join("output", f"{mechanism_name}_validation_runs", timestamp)
  os.makedirs(save_dir, exist_ok=True)

  np.save(os.path.join(save_dir, "mean_diff.npy"),    mean_diff_dict,    allow_pickle=True)
  np.save(os.path.join(save_dir, "eigval_dists.npy"), eigval_dists_dict, allow_pickle=True)
  np.save(os.path.join(save_dir, "coverage.npy"),     coverage_dict,     allow_pickle=True)
  np.save(os.path.join(save_dir, "props_missing.npy"), props_missing)

  print(f"Saved to {save_dir}")
  return mean_diff_dict, eigval_dists_dict, coverage_dict, props_missing, save_dir
