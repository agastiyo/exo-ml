#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

#%%
df = pd.read_csv('data/PS_2026.02.17_12.01.19.csv',comment='#')
df.drop_duplicates(subset='hostname',inplace=True)
# This should be safe, but im not completely sure of it. Might be dropping needed data, but it should only be a few cases with no significant effects

#%%
# Create Matrix X for PCA
Xt = np.array([df['sy_pnum'].to_numpy(),
              df['st_teff'].to_numpy(),
              df['st_rad'].to_numpy(),
              df['st_mass'].to_numpy(),
              df['st_met'].where(df['st_metratio'] == '[Fe/H]').to_numpy(),
              #df['st_met'].where(df['st_metratio'] == '[M/H]').to_numpy(), redundant, too few entries in exoplanet hosts
              10**np.asarray(df['st_lum']),
              10**np.asarray(df['st_logg']),
              df['st_age'].to_numpy()
              #df['st_dens'].to_numpy(), not included in Gaia DR3, too few entries in exoplanet hosts
              #df['st_vsin'].to_numpy(), Too few entries in Gaia DR3 and exoplanet hosts
              #df['st_rotp'].to_numpy(), not included in Gaia DR3, too few entries in exoplanet hosts
              #df['st_radv'].to_numpy(), not included in Gaia DR3, too few entries in exoplanet hosts
              ])

X = Xt.T
print(X)
print(X.shape)
# %%
for i in range(X.shape[1]):
  nancount = 0
  for j in range(4506):
    if np.isnan(X[j][i]):
      nancount += 1
  print(f"{nancount}/4506 NaN in col {i}")
  
# We are now left with matrix X with 4506 entries and 8 features.

# Optimal sample size should be 2^D. For us, we need a minimum of 256 samples for a clear picture. This is promising as our complete dataset is far larger.
# %%
# Try running PCA on a smaller dataset where only rows with all dimensions known are considered.
X_known = X[np.isfinite(X).all(axis=1)]
Xt_known = X_known.T
print(X_known.shape)
# The size of this dataset is also far larger than 256
# %%
# Standardize each column in X_known and Xt_known before PCA
scaler = StandardScaler()
X_known_standard = scaler.fit_transform(X_known)
Xt_known_standard = X_known_standard.T

#%%
# Compute the covariance matrix
C_known = (1 / (X_known_standard.shape[1] - 1)) * (Xt_known_standard @ X_known_standard)

# Perform PCA on the covariance matrix C_known
# Compute eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(C_known)

# Sort eigenvalues and eigenvectors in descending order of eigenvalues
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

feature_names = ['# of Planets', 'T_eff', 'Radius', 'Mass', '[Fe/H]', 'Luminosity', 'Surface Gravity', 'Age']

# Print a summary of each principal component
total_variance = np.sum(eigenvalues)
for i, eigenvalue in enumerate(eigenvalues):
  proportion_variance = eigenvalue / total_variance
  print(f"Principal Component {i + 1}:")
  print(f"  Eigenvalue: {eigenvalue:.4f}")
  print(f"  Proportion of Variance: {proportion_variance:.4%}")
  print("  Contribution of Features:")
  for feature, contribution in zip(feature_names, eigenvectors[:, i]):
    print(f"{feature:>20}: {contribution:>7.4f}")

# %%
plt.scatter(X_known[:,7],X_known[:,0],s=0.1,c='black')
# %%
