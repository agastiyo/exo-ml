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
  'sy_pnum'    : df['sy_pnum'].to_numpy(), # Sample from itself
  'sy_snum'    : df['sy_snum'].to_numpy(), # Sample from itself
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

def synthetic_stellarhosts(n_stars=500, seed=None):
  rng = np.random.default_rng()
  if seed:
    rng = np.random.default_rng(seed=seed)
  
  sy_pnum = rng.poisson(1.5, n_stars) + 1 # Poisson distribution of planets (can't have zero planets since this is a stellar host population)
  sy_snum = rng.poisson(0.1, n_stars) + 1 # Poisson distribution of per system star count (cant have zero stars in the system)
  
  sy_pnum = sy_pnum - (sy_snum-1) # Reduce planets if many stars in the system
  sy_pnum = np.array([1 if i<1 else i for i in sy_pnum]) # Make sure all stars have at least 1 planet
  
  st_teff = rng.choice(np.load(f"{gaia_dir}/teff.npy"), n_stars) # Temperatures drawn from Gaia in kelvin
  st_rad =  rng.choice(np.load(f"{gaia_dir}/radius.npy"), n_stars) # Radius drawn from Gaia in solar radii
  st_mass = rng.choice(np.load(f"{gaia_dir}/mass.npy"), n_stars) # Mass drawn from Gaia in solar masses
  
  st_met_FeH = rng.choice(np.load(f"{gaia_dir}/feh.npy"), n_stars) # Metallicity drawn from Gaia in dex
  st_met_FeH = st_met_FeH + np.abs(0.2*sy_pnum*st_met_FeH) # Increase by 20% of its absolute value for each planet
  
  st_lum = 4*np.pi*((st_rad*696000000)**2)*(5.67e-8)*(st_teff**4) # Luminosity calculated in watts
  st_logg = np.log10(2.74e4 * (st_mass / (st_rad**2))) # log surface gravity calculated in log_10(cm/s^2)
  st_age = rng.choice(np.load(f"{gaia_dir}/age.npy"), n_stars) # Age drawn from Gaia in Gyr
  
  synth_X = np.array([sy_pnum, sy_snum, st_teff, st_rad, st_mass, st_met_FeH, st_lum, st_logg, st_age])
  synth_X = synth_X.T
  
  return synth_X, ['sy_pnum','sy_snum','st_teff','st_rad','st_mass','st_met_FeH','st_lum','st_logg','st_age']