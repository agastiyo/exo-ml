#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
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

# %%
plt.hist(gaia_arrays["st_met_FeH"],bins=480)
plt.yscale('log')
plt.ylabel("log Count")

# %%
