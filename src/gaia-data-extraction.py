#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
gaiaDf = pd.read_csv("data/AstrophysicalParameters_000000-003111.csv",comment='#')
print(gaiaDf)

# %%
plt.hist(gaiaDf['teff_gspphot'],bins=480)
plt.yscale('log')
plt.title(r"Stellar $T_{eff}$ Distribution")
plt.xlabel(r"$T_{eff}$ [K]")
plt.ylabel("log Count")
# %%
plt.hist(gaiaDf['radius_gspphot'],bins=480)
plt.yscale('log')
plt.title(r"Stellar Radius Distribution")
plt.xlabel(r"Radius [$R_{sun}$]")
plt.ylabel("log Count")
# %%
plt.hist(gaiaDf['mass_flame'],bins=480)
plt.yscale('log')
plt.title(r"Stellar Mass Distribution")
plt.xlabel(r"Mass [$M_{sun}$]")
plt.ylabel("log Count")
# %%
plt.hist(gaiaDf['fem_gspspec'].to_numpy() * gaiaDf['mh_gspspec'].to_numpy(),bins=480)
plt.yscale('log')
plt.title(r"Stellar Metallicity Distribution")
plt.xlabel(r"$\frac{Fe}{H}$ [dex]")
plt.ylabel("log Count")
# %%
plt.hist(gaiaDf['mh_gspspec'],bins=480)
plt.yscale('log')
plt.title(r"Stellar Metallicity Distribution")
plt.xlabel(r"$\frac{M}{H}$ [dex]")
plt.ylabel("log Count")
# %%
