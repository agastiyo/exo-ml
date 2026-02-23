import pandas as pd
import numpy as np
import re

# Read the raw NASA Exoplanet Archive export. The file contains many
# rows per system/host coming from different references; we will reduce
# this to one row per host by keeping the most recent/sturdiest entry.
# The file starts with lines beginning with '#', so pass comment="#"
# to `read_csv` to ignore the metadata header.
df = pd.read_csv('data/raw/STELLARHOSTS_2026.02.22_11.35.01.csv', comment='#')
print(df.shape)

# ---------------------------------------------------------------------------
# Helper: extract a publication year (4-digit) from the `st_refname` string.
# If no reasonable year is found, return NaN. We search for years in
# the 1800-2099 range to avoid spurious matches.
def _extract_year(s):
  # Keep pandas NaN values intact
  if pd.isna(s):
    return np.nan
  s = str(s)
  # find 4-digit years in a reasonable range (1800-2099)
  years = re.findall(r"\b(18\d{2}|19\d{2}|20\d{2})\b", s)
  if years:
    # If multiple years appear in the reference string, choose the latest
    return max(int(y) for y in years)
  # No year found
  return np.nan

# Add a convenience column with the extracted year for each reference row
df['st_ref_year'] = df['st_refname'].apply(_extract_year)

# ---------------------------------------------------------------------------
# Columns used to evaluate how 'complete' a stellar-parameter row is.
# When no year is available for any rows of a host, we fall back to
# selecting the row with the most non-null values among these columns.
param_cols = ['st_teff', 'st_rad', 'st_mass', 'st_met', 'st_lum', 'st_logg', 'st_age','st_dens', 'st_vsin', 'st_rotp', 'st_radv']


# Helper: choose the single best row for a grouped host DataFrame.
# Selection strategy:
# 1) If any rows for the host have an extracted `st_ref_year`, restrict
#    candidates to the rows with the maximum (most recent) year.
# 2) Otherwise consider all rows for that host.
# 3) From the candidates, choose the row with the most non-null values
#    in `param_cols`. Ties are broken by selecting the first occurrence.
def _choose_most_recent(group):
  # If at least one row has a year, keep only rows from the latest year
  if group['st_ref_year'].notna().any():
    ymax = group['st_ref_year'].max()
    candidates = group[group['st_ref_year'] == ymax]
  else:
    # No year data available for this host — consider all rows
    candidates = group

  # Count non-null parameter entries per candidate row
  counts = candidates[param_cols].notna().sum(axis=1)
  # `idxmax` returns the index of the first maximum in case of ties
  return candidates.loc[counts.idxmax()]


# ---------------------------------------------------------------------------
# Reduce the DataFrame to a single row per `hostname` using the helper above.
# We preserve the original grouping order by using `sort=False` so selection
# is deterministic when ties occur.
rows = []
for _, g in df.groupby('hostname', sort=False):
  rows.append(_choose_most_recent(g))

# Rebuild DataFrame from the chosen rows and reset index to a clean 0..N-1
df = pd.DataFrame(rows).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Split the metallicity columns based on the used ratio
df['st_met_FeH'] = df['st_met'].where(df['st_metratio'] == '[Fe/H]')
df['st_met_MH'] = df['st_met'].where(df['st_metratio'] == '[M/H]')
df['st_met_NH'] = df['st_met'].where(df['st_metratio'] == '[N/H]')
df['st_met_mH'] = df['st_met'].where(df['st_metratio'] == '[m/H]')
df = df.drop('st_met', axis=1)

print('After keeping most-recent/sturdiest row per host:', df.shape)

# ---------------------------------------------------------------------------
# Save the cleaned table. The output will contain one row per `hostname` and
# include the `st_ref_year` column to document which reference was chosen.
df.to_csv('data/cleaned/STELLARHOSTS.csv', index=False)