import numpy as np
from sklearn.preprocessing import StandardScaler

def RunPCA(X):
  scaler = StandardScaler()
  X_standard = scaler.fit_transform(X)
  
  C = (1 / (X_standard.shape[1] - 1)) * (X_standard.T @ X_standard)
  
  eigvals, eigvecs = np.linalg.eig(C)
  
  sorted_indices = np.argsort(eigvals)[::-1]
  eigvals = eigvals[sorted_indices]
  eigvecs = eigvecs[:, sorted_indices]
  
  return eigvals, eigvecs

def printPCs(eigvals, eigvecs, feature_names = None):
  if not feature_names:
    feature_names = range(len(eigvecs[0]))
  
  assert len(eigvecs[0]) == len(feature_names)
  
  tot_variance = np.sum(eigvals)
  for i,eigval in enumerate(eigvals):
    proportion_variance = eigval / tot_variance
    print(f"Principal Component {i + 1}:")
    print(f"  Eigenvalue: {eigval:.4f}")
    print(f"  Proportion of Variance: {proportion_variance:.4%}")
    print("  Contribution of Features:")
    for feature, contribution in zip(feature_names, eigvecs[:, i]):
      print(f"{feature:>20}: {contribution:>7.4f}")