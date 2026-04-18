import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from scipy.stats import truncnorm

# On its way to depreciation
def EfficientPseudoGibbs(X, X_init,n_iters=15,burn_in=10,n_trees=30,max_depth=10,draws_per_iter=3,weight_str=1,stochastic_str=1,rng=None):
  # Due diligence
  assert X.shape[1] == X_init.shape[1]

  if rng is None:
    rng = np.random.default_rng()

  X = X.copy()
  output = []
  num_features = X.shape[1]

  # Step 1: Create mask array
  isImputed = np.isnan(X)

  # Step 2: Initialize from X_init
  for currCol in range(num_features):
    init_data = X_init[:,currCol]
    init_data = init_data[~np.isnan(init_data)] # Get rid of NaN values

    nanRows = np.isnan(X[:, currCol]) # Mask of nan rows in this col
    N = nanRows.sum() # Number of nan rows in this col

    draws = rng.integers(0, len(init_data), N) # draws from the initial distribution

    X[nanRows,currCol] = init_data[draws] # Initialize the data matrix with the draws
  
  # Step 3: Define the bounds for each feature
  bounds = {}
  for currCol in range(num_features):
    init_data = X_init[:,currCol]
    init_data = init_data[~np.isnan(init_data)]
    
    og_data = X[:,currCol]
    og_data = og_data[~np.isnan(og_data)]
    
    lo = min(np.min(init_data), np.min(og_data))
    hi = max(np.max(init_data), np.max(og_data))
    
    bounds[currCol] = (lo, hi)
  
  # Step 4: Main iteration Loop
  
  # First, define the missing and observed indices for each feature:
  missing_rows = [np.where(isImputed[:, i])[0] for i in range(num_features)]
  observed_rows = [np.where(~isImputed[:, i])[0] for i in range(num_features)]
  
  # Next, define the predictor columns for each feature (X_-p)
  predictor_cols = [np.delete(np.arange(num_features), i) for i in range(num_features)]
  
  # Next, define the random forests for each feature.
  # Each forest gets a seed derived from rng so bootstrap sampling is reproducible.
  forests = [ RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, n_jobs=-1,
                                    random_state=int(rng.integers(0, 2**31)))
              for _ in range(num_features) ]
  
  # Finally, the iteration
  for i in range(n_iters):
    # Train the random forest for each feature
    for p in range(num_features):
      mp = missing_rows[p]
      op = observed_rows[p]
      
      if len(mp) == 0:
        continue
        
      negp = predictor_cols[p]
      
      x_opp = X[op,p]
      
      X_opnegp = X[op][:,negp]
      
      rf = forests[p]
      
      isImputed_opnegp = isImputed[op][:, negp] # This is the mask matrix of X_opnegp
      num_observed = (~isImputed_opnegp).sum(axis=1) # The number of observed values in each row
      row_weights = num_observed / len(negp)
      row_weights = row_weights ** weight_str # Scale the strength of the row weights
      row_weights = np.clip(row_weights, 1e-6, 1) # A weight of 0 won't work, so a very small offset is added. Max should be 1
      
      rf.fit(X_opnegp, x_opp, sample_weight=row_weights)
      
      # The random forest is now trained for this feature!
    
    # Now do the draws
    for k in range(draws_per_iter):
      for p in range(num_features):
        mp = missing_rows[p]
        op = observed_rows[p]
        
        if len(mp) == 0:
          continue
        
        negp = predictor_cols[p]
        
        rf = forests[p]
        
        X_mpnegp = X[mp][:,negp]

        # Per-tree predictions: shape (n_trees, n_missing)
        tree_preds = np.stack([tree.predict(X_mpnegp) for tree in rf.estimators_], axis=0)

        xPredicted_mpp = tree_preds.mean(axis=0)
        sigma_per_row  = tree_preds.std(axis=0) * stochastic_str
        sigma_per_row  = np.maximum(sigma_per_row, 1e-6)

        # The new values are drawn from a truncated normal
        lo, hi = bounds[p]

        a = (lo - xPredicted_mpp) / sigma_per_row
        b = (hi - xPredicted_mpp) / sigma_per_row

        yNew_mpp = truncnorm.rvs(a, b, loc=xPredicted_mpp, scale=sigma_per_row, random_state=rng)
        
        # The new values 
        X[mp,p] = yNew_mpp
      
      # Save to the output
      output.append(X.copy())
    
  # Done with the iteration
  # Remove the burn in iterations
  output = output[burn_in:]
  
  return np.array(output)

# The current algorithm
def EfficientPseudoGibbs_withoutInitializerMatrix(X, bounds_list,n_iters=15,burn_in=10,n_trees=30,max_depth=10,draws_per_iter=3,weight_str=1,stochastic_str=1,rng=None):
  # bounds_list: list of P tuples (lo, hi), either element may be None
  assert len(bounds_list) == X.shape[1]

  if rng is None:
    rng = np.random.default_rng()

  X = X.copy()
  output = []
  num_features = X.shape[1]

  # Step 1: Create mask array
  isImputed = np.isnan(X)

  # Step 2: Initialize missing values
  for currCol in range(num_features):
    nanRows = np.isnan(X[:, currCol])
    N = nanRows.sum()
    if N == 0:
      continue

    lo, hi = bounds_list[currCol]

    if lo is not None and hi is not None:
      X[nanRows, currCol] = rng.uniform(lo, hi, N)
    else:
      obs_vals = X[~nanRows, currCol]
      if len(obs_vals) == 0:
        raise ValueError(f"Feature {currCol} has no observed values — imputation is undefined.")
      fill = float(np.mean(obs_vals))
      if lo is not None:
        fill = max(fill, lo)
      if hi is not None:
        fill = min(fill, hi)
      X[nanRows, currCol] = fill

  # Step 3: Main iteration Loop

  # First, define the missing and observed indices for each feature:
  missing_rows = [np.where(isImputed[:, i])[0] for i in range(num_features)]
  observed_rows = [np.where(~isImputed[:, i])[0] for i in range(num_features)]

  # Next, define the predictor columns for each feature (X_-p)
  predictor_cols = [np.delete(np.arange(num_features), i) for i in range(num_features)]

  # Next, define the random forests for each feature.
  # Each forest gets a seed derived from rng so bootstrap sampling is reproducible.
  forests = [ RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, n_jobs=-1,
                                    random_state=int(rng.integers(0, 2**31)))
              for _ in range(num_features) ]

  # Finally, the iteration
  for i in range(n_iters):
    # Train the random forest for each feature
    for p in range(num_features):
      mp = missing_rows[p]
      op = observed_rows[p]

      if len(mp) == 0:
        continue

      negp = predictor_cols[p]

      x_opp = X[op,p]

      X_opnegp = X[op][:,negp]

      rf = forests[p]

      isImputed_opnegp = isImputed[op][:, negp] # This is the mask matrix of X_opnegp
      num_observed = (~isImputed_opnegp).sum(axis=1) # The number of observed values in each row
      row_weights = num_observed / len(negp)
      row_weights = row_weights ** weight_str # Scale the strength of the row weights
      row_weights = np.clip(row_weights, 1e-6, 1) # A weight of 0 won't work, so a very small offset is added. Max should be 1

      rf.fit(X_opnegp, x_opp, sample_weight=row_weights)

      # The random forest is now trained for this feature!

    # Now do the draws
    for k in range(draws_per_iter):
      for p in range(num_features):
        mp = missing_rows[p]
        op = observed_rows[p]

        if len(mp) == 0:
          continue

        negp = predictor_cols[p]

        rf = forests[p]

        X_mpnegp = X[mp][:,negp]

        # Per-tree predictions: shape (n_trees, n_missing)
        tree_preds = np.stack([tree.predict(X_mpnegp) for tree in rf.estimators_], axis=0)

        xPredicted_mpp = tree_preds.mean(axis=0)
        sigma_per_row  = tree_preds.std(axis=0) * stochastic_str
        sigma_per_row  = np.maximum(sigma_per_row, 1e-6)

        # The new values are drawn from a truncated normal
        lo, hi = bounds_list[p]

        a = (lo - xPredicted_mpp) / sigma_per_row if lo is not None \
            else np.full(len(xPredicted_mpp), -np.inf)
        b = (hi - xPredicted_mpp) / sigma_per_row if hi is not None \
            else np.full(len(xPredicted_mpp), +np.inf)

        yNew_mpp = truncnorm.rvs(a, b, loc=xPredicted_mpp, scale=sigma_per_row, random_state=rng)

        # The new values
        X[mp,p] = yNew_mpp

      # Save to the output
      output.append(X.copy())

  # Done with the iteration
  # Remove the burn in iterations
  output = output[burn_in:]

  return np.array(output)