import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from scipy.stats import truncnorm

def PseudoGibbsImputer(X, X_initializer, regressor:RandomForestRegressor, save_directory,initializer_bins=100,tot_iters=500,burn_in=20,thinning=10,weight_strength=1.0,stochastic_strength=0.25):
  
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
    init_data = X_initializer[:,col]
    init_data = init_data[~np.isnan(init_data)]
    
    og_data = X[:,col]
    og_data = og_data[~np.isnan(og_data)]
    
    lo = min(np.min(init_data), np.min(og_data))
    hi = max(np.max(init_data), np.max(og_data))
    
    bounds[col] = (lo, hi)
  
  # Fourth step:
  # The Random Forest Regression Step
  missing_rows = [np.where(isImputed[:, i])[0] for i in range(num_features)]
  observed_rows = [np.where(~isImputed[:, i])[0] for i in range(num_features)]
  
  predictor_cols = [np.delete(np.arange(num_features), i) for i in range(num_features)]
  
  forests = [ clone(regressor) for _ in range(X.shape[1]) ]

  for i in range(tot_iters):
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
      
      # Training the random forest
      # The weight of each row is determined as (# of observed values / total values in the row)
      # This allows for rows with more observed values to be weighted more than rows with mostly imputed values.
      rf = forests[p]
      
      isImputed_opnegp = isImputed[op][:, negp] # This is the mask matrix of Y_opnegp
      num_observed = (~isImputed_opnegp).sum(axis=1) # The number of observed values in each row
      row_weights = num_observed / len(negp)
      row_weights = row_weights ** weight_strength # Scale the strength of the row weights
      row_weights = np.clip(row_weights, 1e-6, 1) # A weight of 0 won't work, so a very small offset is added. Max should be 1
      
      rf.fit(Y_opnegp, y_opp, sample_weight=row_weights)
      
      yPredicted_mpp = rf.predict(Y_mpnegp)
      
      # Calculate the std using the current imputed values and the prediction by the random forest
      # The std is then scaled by the defined stochastic strength of the algorithm
      # A std of 0 won't work, so a very small offset is added
      sigma = np.sqrt(np.mean( (y_mpp - yPredicted_mpp) ** 2 ))
      sigma *= stochastic_strength
      sigma = max(sigma, 1e-6)
      
      # The new values are drawn from a truncated normal
      lo, hi = bounds[p]
      
      a = (lo - yPredicted_mpp) / sigma
      b = (hi - yPredicted_mpp) / sigma

      yNew_mpp = truncnorm.rvs(a, b, loc=yPredicted_mpp, scale=sigma)
      
      # The new values 
      X[mp,p] = yNew_mpp
      
      sqdiff.append( (X[mp,p] - y_mpp) ** 2 )
    
    rmse_hist.append(np.sqrt(np.mean(np.concatenate(sqdiff))))
    
    if save_directory:
      if (i+1)%thinning == 0 and (i+1) > burn_in:
        np.save(f"{save_directory}/imputed_iter_{i+1}.npy", X)
    
    print(f"Iteration {i+1}/{tot_iters} done")
  
  return X,rmse_hist

def EfficientPseudoGibbs(X, X_init,n_iters=15,burn_in=10,n_trees=30,max_depth=10,draws_per_iter=3,weight_str=1,stochastic_str=0.25):
  # Due diligence
  assert X.shape[1] == X_init.shape[1]
  
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
    
    draws = np.random.randint(0,len(init_data),N) # draws from the initial distribution
    
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
  
  # Next, define the random forests for each feature
  forests = [ clone(RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, n_jobs=-1)) for _ in range(num_features) ]
  
  # Finally, the iteration
  for i in range(n_iters):
    print(f"Iteration {i+1}:")
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
    
    print("    Trained!")
    
    # Now do the draws
    for k in range(draws_per_iter):
      for p in range(num_features):
        mp = missing_rows[p]
        op = observed_rows[p]
        
        if len(mp) == 0:
          continue
        
        negp = predictor_cols[p]
        
        rf = forests[p]
        
        x_mpp = X[mp,p]
        X_mpnegp = X[mp][:,negp]
        
        xPredicted_mpp = rf.predict(X_mpnegp)
        
        # Standard deviation scaled by stochastic strength
        sigma = np.sqrt(np.mean( (x_mpp - xPredicted_mpp) ** 2 ))
        sigma *= stochastic_str
        sigma = max(sigma, 1e-6)
        
        # The new values are drawn from a truncated normal
        lo, hi = bounds[p]
        
        a = (lo - xPredicted_mpp) / sigma
        b = (hi - xPredicted_mpp) / sigma

        yNew_mpp = truncnorm.rvs(a, b, loc=xPredicted_mpp, scale=sigma)
        
        # The new values 
        X[mp,p] = yNew_mpp
      
      # Save to the output
      output.append(X.copy())
    
      print(f"    {k+1}/{draws_per_iter} draws complete")
    
  # Done with the iteration
  # Remove the burn in iterations
  output = output[burn_in:]
  
  return output