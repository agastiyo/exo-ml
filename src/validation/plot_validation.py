#%%
import numpy as np
import os
import matplotlib.pyplot as plt

# Path to the timestamped run folder to plot.
# e.g. "output/mcar_validation_runs/20260415_143022"
run_dir = "output/mcar_validation_runs/REPLACE_WITH_TIMESTAMP"

#%%
mean_diff_dict    = np.load(os.path.join(run_dir, "mean_diff.npy"),    allow_pickle=True).item()
eigval_dists_dict = np.load(os.path.join(run_dir, "eigval_dists.npy"), allow_pickle=True).item()
coverage_dict     = np.load(os.path.join(run_dir, "coverage.npy"),     allow_pickle=True).item()
props_missing     = np.load(os.path.join(run_dir, "props_missing.npy"))

method_names            = list(mean_diff_dict.keys())
stochastic_method_names = list(coverage_dict.keys())

# Derive a title prefix from the folder path, e.g. "MCAR" from "mcar_validation_runs"
folder_name   = os.path.basename(os.path.dirname(run_dir))   # "mcar_validation_runs"
mechanism_tag = folder_name.replace("_validation_runs", "").upper()

#%%
# --- RMSE performance plot ---

plt.figure(figsize=(10, 6))

for name in method_names:
  data  = np.array(mean_diff_dict[name])   # shape: (n_tot, n_props)
  means = np.mean(data, axis=0)
  stds  = np.std(data, axis=0)
  plt.errorbar(props_missing, means, yerr=stds, marker='o', label=name)

plt.xlabel("Fraction of Missing Data", fontsize=12)
plt.ylabel("RMSE (on masked entries)", fontsize=12)
plt.title(f"{mechanism_tag} Imputation Performance", fontsize=14)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "performance.png"))
plt.show()

#%%
# --- Eigenvalue spectrum log-L2 plot ---

plt.figure(figsize=(10, 6))

for name in method_names:
  data  = np.array(eigval_dists_dict[name])
  means = np.mean(data, axis=0)
  stds  = np.std(data, axis=0)
  plt.errorbar(props_missing, means, yerr=stds, marker='o', label=name)

plt.xlabel("Fraction of Missing Data", fontsize=12)
plt.ylabel("Log L2 dist from true Eigval spectra", fontsize=12)
plt.title(f"{mechanism_tag} Pattern Destruction", fontsize=14)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "pattern.png"))
plt.show()

#%%
# --- IQR coverage plot (stochastic methods only) ---
# Ideal calibration: the true masked values fall inside the 25-75 percentile
# range of the ensemble 50% of the time, so a horizontal line at 0.5 is drawn
# for reference. Lower = over-confident, higher = under-confident.

plt.figure(figsize=(10, 6))

for name in stochastic_method_names:
  data  = np.array(coverage_dict[name])
  means = np.mean(data, axis=0)
  stds  = np.std(data, axis=0)
  plt.errorbar(props_missing, means, yerr=stds, marker='o', label=name)

plt.axhline(0.5, color='k', linestyle='--', linewidth=1, alpha=0.6, label='Ideal (0.5)')
plt.xlabel("Fraction of Missing Data", fontsize=12)
plt.ylabel("IQR Coverage of True Values", fontsize=12)
plt.title(f"{mechanism_tag} Uncertainty Calibration", fontsize=14)
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "coverage.png"))
plt.show()

# %%
