#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#%%
df = pd.read_csv('data/cleaned/STELLARHOSTS.csv',comment='#')
print(df.shape)

#%%
# Check how many NaN values are in each column
nanCheck = ['sy_pnum','sy_snum','st_teff','st_rad','st_mass','st_met_FeH','st_met_MH','st_met_NH','st_met_mH','st_lum','st_logg','st_age','st_dens','st_vsin','st_rotp','st_radv']

for col in nanCheck:
  nancount = 0
  for entry in df[col]:
    if np.isnan(entry):
      nancount += 1
  percent = (nancount/len(df[col]))
  print(f"{col:>10}: {nancount:>4} NaN ({percent:.2%})")

#%%
# Create Matrix X for PCA
Xt = np.array([df['sy_pnum'].to_numpy(),
              df['sy_snum'].to_numpy(),
              df['st_teff'].to_numpy(),
              df['st_rad'].to_numpy(),
              df['st_mass'].to_numpy(),
              df['st_met_FeH'].to_numpy(),
              #df['st_met_MH'].to_numpy(), redundant, too few entries in exoplanet hosts
              #df['st_lum'].to_numpy(),
              df['st_logg'].to_numpy(),
              df['st_age'].to_numpy(),
              df['st_dens'].to_numpy()
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
  for j in range(X.shape[0]):
    if np.isnan(X[j][i]):
      nancount += 1
  print(f"{nancount}/{X.shape[0]} NaN in col {i}")

# Optimal sample size should be 2^D.
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

feature_names = ['# of Planets', '# of Stars', 'T_eff', 'Radius', 'Mass', 'Fe/H','log g','age','density']

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
