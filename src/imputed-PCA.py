#%%
import numpy as np
import glob
import src.utils.pca as PCA

#%%
files = sorted(glob.glob("data/imputed/imputed_iter_*.npy"))
samples = [np.load(f) for f in files]

stack = np.stack(samples)

components = []
explained = []

for X in samples:
  eigvals, eigvecs = PCA.RunPCA(X)

  var_ratio = eigvals / np.sum(eigvals)

  components.append(eigvecs.T)
  explained.append(var_ratio)

# Convert to arrays
components = np.stack(components)
explained = np.stack(explained)

# --- Align component signs to first solution ---
ref = components[0]

for i in range(1, components.shape[0]):
    for k in range(components.shape[1]):
        if np.dot(components[i, k], ref[k]) < 0:
            components[i, k] *= -1

# --- Combine results across imputations ---

# Mean PCA loadings
mean_components = components.mean(axis=0)
std_components = components.std(axis=0)

# Mean explained variance
mean_explained = explained.mean(axis=0)
std_explained = explained.std(axis=0)

print("Mean explained variance:")
print(mean_explained)

print("\nStd of explained variance across imputations:")
print(std_explained)
# %%
# Recover eigenvalues from explained variance ratios
# (mean_explained = eigenvalue / total_variance)
total_variance = 1.0
mean_eigenvalues = mean_explained * total_variance
std_eigenvalues = std_explained * total_variance

# Print PCA summary
# Make sure that the feature names are in the correct order
feature_names = [
  'sy_pnum','sy_snum',
  'st_teff','st_rad','st_mass','st_met_FeH',
  'st_lum','st_logg','st_age'
]

for i in range(components.shape[1]):
  print(f"\nPrincipal Component {i+1}")
  print("-"*40)

  print(f"Mean Eigenvalue: {mean_eigenvalues[i]:.6f} ± {std_eigenvalues[i]:.6f}")
  print(f"Mean Proportion of Variance: {mean_explained[i]:.4%} ± {std_explained[i]:.4%}")

  print("\nContribution of Features (mean ± std):")

  for j, feature in enumerate(feature_names):
    mean_loading = mean_components[i, j]
    std_loading = std_components[i, j]

    print(f"{feature:>20}: {mean_loading:>8.4f} ± {std_loading:>7.4f}")
# %%
