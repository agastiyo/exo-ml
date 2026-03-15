#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#%%
df = pd.read_csv('data/cleaned/STELLARHOSTS.csv',comment='#')

# Check how many NaN values are in each column
nanCheck = ['sy_pnum','sy_snum','st_teff','st_rad','st_mass','st_met_FeH','st_met_MH','st_met_NH','st_met_mH','st_lum','st_logg','st_age','st_dens','st_vsin','st_rotp','st_radv']
nanDict = {}

for col in nanCheck:
  nancount = 0
  for entry in df[col]:
    if np.isnan(entry):
      nancount += 1
  percent = (nancount/len(df[col]))
  nanDict[col] = percent
  print(f"{col:>10}: {nancount:>4} NaN ({percent:.2%})")

#%%
# Set parameters for running the small sample PCA
tau = 0.7 # nan limit (col can have this percent of nans and still be used)

# Create Matrix X for PCA
Xt = []
for key in nanDict.keys():
  if nanDict[key] <= tau:
    Xt.append(df[key])

Xt = np.array(Xt)
X = Xt.T

# Try running PCA on a smaller dataset where only rows with all dimensions known are considered.
X_known = X[np.isfinite(X).all(axis=1)]
Xt_known = X_known.T
print(X_known.shape)
print(f"Optimal sample size for this dataset is {2**X_known.shape[1]}.")
print(f"tau = {tau} yields {X_known.shape[0]} complete rows")
if X_known.shape[0] == 0:
  raise ValueError("No complete data to conduct PCA on!")

# %%
# Standardize each column in X_known and Xt_known before PCA
scaler = StandardScaler()
X_known_standard = scaler.fit_transform(X_known)
Xt_known_standard = X_known_standard.T

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
