#%%
import numpy as np
import src.utils.datamatrix as DMatrix
from src.validation._common import run_validation_rshm
import warnings

warnings.filterwarnings("ignore")

#%%
X_known, feature_names = DMatrix.synthetic_stellarhosts(n_stars=500)
print(f"Synthetic population shape: {X_known.shape}")
print(f"Features: {feature_names}")

# %%
# Feature order: [sy_pnum(0), sy_snum(1), st_teff(2), st_rad(3), st_mass(4),
#                 st_met_FeH(5), st_lum(6), st_logg(7), st_age(8)]
bounds_list = [
  (1, None),      # sy_pnum
  (1, None),      # sy_snum
  (0, None),      # st_teff  [K]
  (0, None),      # st_rad   [R_sun]
  (0, None),      # st_mass  [M_sun]
  (None, None),   # st_met_FeH [dex]
  (0, None),      # st_lum   [W]
  (0, None),      # st_logg  [log10 cm/s^2]
  (0, 13.8),      # st_age   [Gyr]
]

#%%
def RSHM_mask(X_known, prop_missing, rng,
              alpha_lum=0.30,
              alpha_teff=0.20,
              beta_transit=0.40,
              beta_rv=0.35,
              delta_age=0.30):
    """
    Realistic Stellar Host Missingness (RSHM) mask generator.

    Simulates the detection-driven selection effects that produce MNAR
    missingness in confirmed exoplanet host star catalogs. Five sequential
    steps build a per-feature probability matrix, then use it to (a) fully
    delete rows whose planets would never be detected, and (b) draw
    feature-level cell missingness for the surviving rows.
    """
    N, P = X_known.shape

    # Feature indices
    IDX_TEFF = 2
    IDX_RAD  = 3
    IDX_MET  = 5
    IDX_LUM  = 6

    # Transit detection primarily determines photometric characterization quality
    PHOTOMETRIC_COLS   = [IDX_TEFF, IDX_RAD, IDX_LUM, 7]  # 7 = st_logg
    # RV detection primarily determines spectroscopic characterization quality
    SPECTROSCOPIC_COLS = [IDX_TEFF, IDX_MET]
    # Columns that can ever be masked (sy_pnum, sy_snum excluded)
    MASKABLE_COLS = list(range(2, P))

    def _zscore(col):
        return (col - np.mean(col)) / (np.std(col) + 1e-10)

    # ------------------------------------------------------------------
    # Initialize probability matrix — base probability 0.5 for all maskable
    # features; sy_pnum and sy_snum fixed at 0 (never missing).

    P_mat = np.full((N, P), 0.5)
    P_mat[:, 0:2] = 0.0

    # ------------------------------------------------------------------
    # Step 1 — Survey selection
    # Malmquist bias: magnitude-limited surveys preferentially include
    # luminous stars; sub-luminous stars are underrepresented and poorly
    # characterized. Low z_lum → high row missingness.
    # Teff noise proxy: hotter stars have higher photometric noise (σ_CDPP),
    # reducing detection probability and downstream characterization quality.
    # Low z_teff → high row missingness.
    # Both effects act on the full row (all features).

    z_lum  = _zscore(X_known[:, IDX_LUM])
    z_teff = _zscore(X_known[:, IDX_TEFF])

    row_delta = alpha_lum * (-z_lum) + alpha_teff * (z_teff)  # (N,)
    P_mat[:, MASKABLE_COLS] += row_delta[:, None]

    # ------------------------------------------------------------------
    # Step 2 — Detection method: Transit and RV
    # Transit favors high-luminosity, small-radius stars (deep transits,
    # photometric precision). Well-characterized transit hosts have accurate
    # photometric parameters (Teff, radius, luminosity, logg).
    # RV favors high-luminosity, metal-rich stars (spectroscopic precision,
    # planet-metallicity correlation). Well-characterized RV hosts have
    # accurate spectroscopic parameters (metallicity, Teff).
    # A positive detection score reduces missingness for the relevant
    # feature subset; negative scores are clipped to zero (no penalty).

    z_rad = _zscore(X_known[:, IDX_RAD])
    z_met = _zscore(X_known[:, IDX_MET])

    transit_score = z_lum - z_rad          # high lum, low rad → transit-favorable
    rv_score      = z_lum + z_met          # high lum, high met → RV-favorable

    P_mat[:, PHOTOMETRIC_COLS]   -= beta_transit * transit_score[:, None]
    P_mat[:, SPECTROSCOPIC_COLS] -= beta_rv      * rv_score[:, None]

    # ------------------------------------------------------------------
    # Step 3 — Non-detection row deletion
    # Stars whose average feature missingness probability is highest are
    # fully masked: their planets would never have been detected by transit
    # or RV surveys, so they do not appear in STELLARHOSTS at all.
    # n_rows_delete = round(prop_missing * N) so that deleted rows account
    # for the bulk of the total missing-cell budget.

    n_rows_delete = int(np.floor(prop_missing * N))
    row_avg = P_mat[:, MASKABLE_COLS].mean(axis=1)

    # argpartition is O(N) — faster than full sort for large N
    delete_idx = np.argpartition(row_avg, -n_rows_delete)[-n_rows_delete:]

    mask = np.zeros((N, P), dtype=bool)
    
    mask = np.delete(mask, delete_idx, axis=0)
    P_mat = np.delete(P_mat, delete_idx, axis=0)
    X_known = np.delete(X_known, delete_idx, axis=0)
    N -= n_rows_delete

    # ------------------------------------------------------------------
    # Step 4 — Feature-level cell missingness for retained rows
    # Draw prop_missing * N * P cells from the surviving rows,
    # weighted by their per-feature probabilities.

    n_total_target = int(np.floor(prop_missing * N * P))

    p_flat = np.clip(P_mat.flatten(), 0.0, None)
    p_flat = p_flat / p_flat.sum()

    flat_indices = rng.choice(
        int(N * P),
        size=n_total_target,
        replace=False,
        p=p_flat
    )
    
    rows, cols = np.unravel_index(flat_indices, (N, P))
    mask[rows, cols] = True

    X_masked = X_known.copy()
    X_masked[mask] = np.nan
    return X_masked, mask

#%%
run_validation_rshm(RSHM_mask, X_known, bounds_list, n_tot=1, n_runs=35)

# %%
