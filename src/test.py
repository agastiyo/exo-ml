#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%
df = pd.read_csv('data/PS_2026.02.17_12.01.19.csv',comment='#')
df.drop_duplicates(subset='hostname',inplace=True)
# This should be safe, but im not completely sure of it. Might be dropping needed data, but it should only be a few cases with no significant effects

print(df)

#%%
# Create Matrix X for PCA
Xt = np.array([df['sy_pnum'].to_numpy(),
              df['st_teff'].to_numpy(),
              df['st_rad'].to_numpy(),
              df['st_mass'].to_numpy(),
              df['st_met'].where(df['st_metratio'] == '[Fe/H]').to_numpy(),
              df['st_met'].where(df['st_metratio'] == '[M/H]').to_numpy(),
              10**np.asarray(df['st_lum']),
              10**np.asarray(df['st_logg']),
              df['st_age'].to_numpy(),
              df['st_dens'].to_numpy(),
              df['st_vsin'].to_numpy(),
              df['st_rotp'].to_numpy(),
              df['st_radv'].to_numpy()
              ])

X = Xt.T
print(X)
# %%
for i in range(13):
  nancount = 0
  for j in range(4506):
    if np.isnan(X[j][i]):
      nancount += 1
  print(f"{nancount}/4506 NaN in col {i}")
# %%
