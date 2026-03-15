import pandas as pd
import numpy as np
import os

gaia_path = "data/raw/AstrophysicalParameters_000000-003111.csv"
out_dir = "data/cleaned/gaia_arrays"

os.makedirs(out_dir, exist_ok=True)

# columns needed
cols = {
  "teff": "teff_gspphot",
  "radius": "radius_gspphot",
  "mass": "mass_flame",
  "fem": "fem_gspspec",
  "mh": "mh_gspspec",
  "lum": "lum_flame",
  "logg": "logg_msc1",
  "age": "age_flame",
  "vsini": "vsini_esphs"
}

df = pd.read_csv(gaia_path, usecols=cols.values(), comment="#", low_memory=False)

# metallicity combination
feh = (df["fem_gspspec"] * df["mh_gspspec"]).dropna().to_numpy()

np.save(f"{out_dir}/feh.npy", feh.astype(np.float32))

for key, col in cols.items():
  if col in ["fem_gspspec", "mh_gspspec"]:
    continue

  data = df[col].dropna().to_numpy().astype(np.float32)

  np.save(f"{out_dir}/{key}.npy", data)

print("Saved Gaia arrays to", out_dir)