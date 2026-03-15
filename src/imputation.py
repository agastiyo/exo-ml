#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

#%%
df = pd.read_csv('data/cleaned/STELLARHOSTS.csv',comment='#') # Stellarhost dataframe
gaia_dir = "data/cleaned/gaia_arrays"

gaia_arrays = {
  "st_teff": np.load(f"{gaia_dir}/teff.npy"),
  "st_rad": np.load(f"{gaia_dir}/radius.npy"),
  "st_mass": np.load(f"{gaia_dir}/mass.npy"),
  "st_met_FeH": np.load(f"{gaia_dir}/feh.npy"),
  "st_lum": np.load(f"{gaia_dir}/lum.npy"),
  "st_logg": np.load(f"{gaia_dir}/logg.npy"),
  "st_age": np.load(f"{gaia_dir}/age.npy"),
  "st_vsin": np.load(f"{gaia_dir}/vsini.npy")
}

X_colNames = gaia_arrays.keys()
feature_cols = ['sy_pnum', 'sy_snum', *X_colNames]
  
#%%
# First, we must create a mask array that stores which values were observed and which were missing
# False = observed, True = missing
X = df[feature_cols].to_numpy()
isImputed = np.isnan(X)

# %%
# Next, we must initialize the missing values using the real Gaia distribution of stellar parameters.
df_imputed = df.copy()

for col in X_colNames:
  data = gaia_arrays[col]

  hist, edges = np.histogram(data, bins=480)
  pdf = hist / hist.sum()

  mask = df[col].isna()
  N = mask.sum()

  chosen_bins = np.random.choice(len(hist), size=N, p=pdf)
  u = np.random.rand(N)

  draws = edges[chosen_bins] + u * (edges[chosen_bins+1] - edges[chosen_bins])

  df_imputed.loc[mask, col] = draws

# We can now create the data matrix X
X = df_imputed[feature_cols].to_numpy(dtype=np.float32)

#%%
# We now have a data matrix X with missing values initilaized from the Gaia distribution.
# We also have a mask isImputed that stores which values are original and which are imputed.
# With these, we can go ahead and implement the regression step
n_iters = 50
rmse_hist = []

# Define the ranges to constrain the predicted draws to
bounds = {}
for col in X_colNames:
  data = gaia_arrays[col]

  lo = np.percentile(data, 0.5)
  hi = np.percentile(data, 99.5)

  bounds[col] = (lo, hi)

clip_bounds = {}

clip_bounds[0] = (df['sy_pnum'].min(), df['sy_pnum'].max())
clip_bounds[1] = (df['sy_snum'].min(), df['sy_snum'].max())

for i, col in enumerate(X_colNames):
  clip_bounds[i+2] = bounds[col]
  
# Find out the indexes of observed and missing data for each col
missing_rows = []
observed_rows = []

for col in range(X.shape[1]):
  missing_rows.append(np.where(isImputed[:, col])[0])
  observed_rows.append(np.where(~isImputed[:, col])[0])
  
# Compute the indices used as predictors
predictor_cols = []

for col in range(X.shape[1]):
  predictor_cols.append([i for i in range(X.shape[1]) if i != col])

# Create the random forest regressors
forests = [
  RandomForestRegressor(n_estimators=50, n_jobs=-1)
  for _ in range(X.shape[1])
]

for i in range(n_iters):
  X_old = X.copy()
  
  for col in range(X.shape[1]):
    miss = missing_rows[col]
    obs = observed_rows[col]
    
    if len(miss) == 0:
      continue
    
    pred_cols = predictor_cols[col]

    # Column vector of observed values in the current col
    y_obs = X[obs, col]
    
    # Column vector of imputed values in the current col
    y_old = X[miss, col]
    
    # Matrix of all the other features corresponding to observed values in the current col
    X_obs = X[obs][:, pred_cols]
    
    # Matrix of all the other features corresponding to missing values in the current col
    X_miss = X[miss][:, pred_cols]
    
    # Train the Random Forest Regressor
    rf = forests[col]
    rf.fit(X_obs, y_obs)
    
    # Predict missing rows
    y_pred = rf.predict(X_miss)
    
    # RMSD variance estimate
    s2 = np.mean((y_old - y_pred) ** 2)
    sigma = np.sqrt(s2)

    # Draw and clip stochastic updates
    y_new = np.random.normal(loc=y_pred, scale=sigma)
    lo, hi = clip_bounds[col]
    y_new = np.clip(y_new, lo, hi)

    # Update matrix
    X[miss, col] = y_new
  
  diff = X - X_old
  diff = diff[isImputed]
  rmse_hist.append(np.sqrt(np.mean(diff**2)))
  
  print(f"Iteration {i+1}/{n_iters} done")

# %%
# Convergence analysis of the regression step
plt.plot(np.arange(1,n_iters+1),rmse_hist)
plt.title("Root Mean Squared Error between iterations")

# %%
