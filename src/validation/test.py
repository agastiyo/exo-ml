#%%
import numpy as np
import matplotlib.pyplot as plt
from src.utils.impute import EfficientPseudoGibbs
import src.utils.datamatrix as DMatrix

#%%
tau = 0.7

X = DMatrix.X(tau)
X_init = DMatrix.X_init(tau)
feature_cols = DMatrix.feature_names(tau)

X_output = EfficientPseudoGibbs(X,X_init,n_trees=200)

#%%
X_stack = np.stack(X_output, axis=0)
X_mean_imputed = np.mean(X_stack, axis=0)

n_features = X.shape[1]

for i in range(n_features):
  plt.figure()
  
  # Original data histogram
  plt.hist(X[:, i], bins=50, alpha=0.5, label='Original', density=True)
  
  # Imputed mean histogram
  plt.hist(X_mean_imputed[:, i], bins=50, alpha=0.5, label='Imputed (Mean of samples)', density=True)
  
  plt.title(f"Feature: {feature_cols[i]}")
  plt.xlabel("Value")
  plt.ylabel("Density")
  plt.legend()
  
  plt.show()
# %%
