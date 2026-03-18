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

X_orig = df[feature_cols].to_numpy()

#%%
it = '355'
X_imp = np.load(f"data/imputed/imputed_iter_{it}.npy")

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
dataset_range = np.arange(15,55,5)

for i in dataset_range:
  X_imp = np.load(f"data/imputed/imputed_iter_{i}.npy")
  cell_hist.append(X_imp[0][7])

plt.plot(cell_hist)
print(cell_hist)

# %%
