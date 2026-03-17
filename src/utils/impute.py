import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone

def PseudoGibbsImputer(X, X_initializer, regressor:RandomForestRegressor, save_directory, n_iters=20, save_every=5, initializer_bins=100):
  
  X = X.copy()
  
  assert X.shape[1] == X_initializer.shape[1]
  
  num_features = X.shape[1]
  rmse_hist = []
  
  # First step:
  # Create the mask array
  isImputed = np.isnan(X)
  
  # Second step:
  # Initialize missing values using the initializer matrix
  for currCol in range(num_features):
    initializer_data = X_initializer[:,currCol]
    initializer_data = initializer_data[~np.isnan(initializer_data)]
    
    hist, edges = np.histogram(initializer_data, bins=initializer_bins)
    pdf = hist / hist.sum()
    
    nanRows = np.isnan(X[:, currCol])
    N = nanRows.sum()
    
    chosen_bins = np.random.choice(len(hist), size=N, p=pdf)
    u = np.random.rand(N)
    
    draws = edges[chosen_bins] + u * (edges[chosen_bins+1] - edges[chosen_bins])
    
    X[nanRows,currCol] = draws
  
  # Third step:
  # Define the ranges to constrain predicted draws in
  bounds = {}
  for col in range(num_features):
    data = X_initializer[:,col]
    data = data[~np.isnan(data)]
    
    lo = np.percentile(data, 0.5)
    hi = np.percentile(data, 99.5)
    
    bounds[col] = (lo, hi)
  
  # Fourth step:
  # The Random Forest Regression Step
  missing_rows = [np.where(isImputed[:, i])[0] for i in range(num_features)]
  observed_rows = [np.where(~isImputed[:, i])[0] for i in range(num_features)]
  
  predictor_cols = [np.delete(np.arange(num_features), i) for i in range(num_features)]
  
  forests = [ clone(regressor) for _ in range(X.shape[1]) ]
  
  for i in range(n_iters):
    sqdiff = []
    
    for p in range(num_features):
      mp = missing_rows[p]
      op = observed_rows[p]
      
      if len(mp) == 0:
        continue
      
      negp = predictor_cols[p]
      
      y_mpp = X[mp,p]
      y_opp = X[op,p]
      
      Y_mpnegp = X[mp][:,negp]
      Y_opnegp = X[op][:,negp]
      
      rf = forests[p]
      rf.fit(Y_opnegp,y_opp)
      
      yPredicted_mpp = rf.predict(Y_mpnegp)
      
      sigma = np.sqrt(np.mean( (y_mpp - yPredicted_mpp) ** 2 ))
      sigma = max(sigma, 1e-6)
      
      yNew_mpp = np.random.normal(loc=yPredicted_mpp, scale=sigma)
      lo, hi = bounds[p]
      yNew_mpp = np.clip(yNew_mpp, lo, hi)
      
      X[mp,p] = yNew_mpp
      
      sqdiff.append( (X[mp,p] - y_mpp) ** 2 )
    
    rmse_hist.append(np.sqrt(np.mean(np.concatenate(sqdiff))))
    
    if (i + 1) % save_every == 0:
      np.save(f"{save_directory}/imputed_iter_{i+1}.npy", X)
    
    print(f"Iteration {i+1}/{n_iters} done")
  
  return rmse_hist