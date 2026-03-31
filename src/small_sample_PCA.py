#%%
import pandas as pd
import numpy as np
import src.utils.pca as PCA
import src.utils.datamatrix as DMatrix

#%%
tau = 0.7

X = DMatrix.X(tau)

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
feature_names = DMatrix.feature_names(tau)
PCA.printPCs(eigvals, eigvecs, feature_names)
# %%
