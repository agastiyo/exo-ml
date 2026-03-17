#%%
import pandas as pd
import numpy as np
import src.utils.pca as PCA

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
print(X_known.shape)
print(f"Optimal sample size for this dataset is {2**X_known.shape[1]}.")
print(f"tau = {tau} yields {X_known.shape[0]} complete rows")
if X_known.shape[0] == 0:
  raise ValueError("No complete data to conduct PCA on!")

# %%
# Run PCA
eigvals, eigvecs = PCA.RunPCA(X_known)

# Print PCs
feature_names = ['# of Planets', '# of Stars', 'T_eff', 'Radius', 'Mass', 'Fe/H','log g','age','density']
PCA.printPCs(eigvals, eigvecs, feature_names)

# %%
