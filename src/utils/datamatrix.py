import pandas as pd
import numpy as np

df = pd.read_csv('data/cleaned/STELLARHOSTS.csv',comment='#')

features = ['sy_pnum','sy_snum','st_teff','st_rad','st_mass','st_met_FeH','st_met_MH','st_met_NH','st_met_mH','st_lum','st_logg','st_age','st_dens','st_vsin','st_rotp','st_radv']

# Create the dictionary that stores the missingness of each feature
percentMissing = {}

for col in features:
  nancount = 0
  for entry in df[col]:
    if np.isnan(entry):
      nancount += 1
  percent = (nancount/len(df[col]))
  percentMissing[col] = percent

# Create the dictionary that stores the corresponding initial distributions
gaia_dir = "data/cleaned/gaia_arrays"
initDict = {
  'sy_pnum'    : np.ones(1), # Complete row, so this initializer doesn't matter
  'sy_snum'    : np.ones(1), # Complete row, so this initializer doesn't matter
  'st_teff'    : np.load(f"{gaia_dir}/teff.npy"),
  'st_rad'     : np.load(f"{gaia_dir}/radius.npy"),
  'st_mass'    : np.load(f"{gaia_dir}/mass.npy"),
  'st_met_FeH' : np.load(f"{gaia_dir}/feh.npy"),
  'st_met_MH'  : df['st_met_MH'].to_numpy(), # No corresponding gaia feature, sample from itself
  'st_met_NH'  : df['st_met_NH'].to_numpy(), # No gaia feature
  'st_met_mH'  : df['st_met_mH'].to_numpy(), # No gaia feature
  'st_lum'     : np.load(f"{gaia_dir}/lum.npy"),
  'st_logg'    : np.load(f"{gaia_dir}/logg.npy"),
  'st_age'     : np.load(f"{gaia_dir}/age.npy"),
  'st_dens'    : df['st_dens'].to_numpy(), # No gaia feature
  'st_vsin'    : np.load(f"{gaia_dir}/age.npy"),
  'st_rotp'    : df['st_rotp'].to_numpy(), # No gaia feature
  'st_radv'    : df['st_radv'].to_numpy() # No gaia feature
}

def X(tau=0.7):
  '''Creates the Data matrix X using the STELLARHOST data and the given completeness threshold.
  
  Only features with a missing fraction less than the threshold are included.
  '''
  inc_features = []
  
  for key,val in percentMissing.items():
    if val <= tau:
      inc_features.append(key)
  
  return df[inc_features].to_numpy()

def X_init(tau=0.7):
  '''Creates the initializer matrix X_init given the completeness threshold.
  
  Only features with a missing fraction less than the threshold are included.
  '''
  inc_features = []
  
  for key,val in percentMissing.items():
    if val <= tau:
      inc_features.append(key)
  
  rows = []
  
  for feature in inc_features:
    rows.append(initDict[feature])
  
  # Pad the length of all rows to standardize size
  max_len = max(r.size for r in rows)

  X_init = np.full((len(rows), max_len), np.nan, dtype=float)

  for i, r in enumerate(rows):
    X_init[i, :r.size] = r

  X_init = X_init.T
  
  return X_init

def feature_names(tau=0.7):
  inc_features = []
  
  for key,val in percentMissing.items():
    if val <= tau:
      inc_features.append(key)
  
  return inc_features