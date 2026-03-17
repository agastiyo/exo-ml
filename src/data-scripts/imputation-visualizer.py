#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
df = pd.read_csv('data/cleaned/STELLARHOSTS.csv',comment='#') # Stellarhost dataframe
gaia_dir = "data/cleaned/gaia_arrays"

feature_cols = [
  'sy_pnum','sy_snum',
  'st_teff','st_rad','st_mass','st_met_FeH',
  'st_lum','st_logg','st_age'
]

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

X_orig = df[feature_cols].to_numpy()

#%%
it = '490'
X_imp = np.load(f"data/imputed/imputed_iter_{it}.npy")

#%%
fig, axes = plt.subplots(9, 1, figsize=(7, 25))

for i, ax in enumerate(axes.flatten()):
    
  feature = feature_cols[i]

  ax.hist(X_orig[:, i], bins=50, alpha=0.5, label='original', density=True)
  ax.hist(X_imp[:, i], bins=50, alpha=0.5, label='imputed', density=True)

  ax.set_title(f"{feature} at iteration {it}")
  ax.legend()

plt.tight_layout()
plt.show()
#%%
cell_hist = []
dataset_range = np.arange(10,510,10)

for i in dataset_range:
  X_imp = np.load(f"data/imputed/imputed_iter_{i}.npy")
  cell_hist.append(X_imp[40][4])

plt.plot(cell_hist)
print(cell_hist)

# %%
