# SWRF-Impute: Stochastic Weighted Random Forest Imputation for Stellar Host Data

## Overview

**SWRF-Impute** is an imputation method designed to handle missing data in stellar host datasets while preserving the statistical structure of the data and providing calibrated uncertainty estimates. This project demonstrates that SWRF-Impute achieves comparable reconstruction accuracy to state-of-the-art methods while respecting physical constraints and enabling more reliable downstream astrophysical inference for stellar host population analysis.

## Key Contributions

✓ **Comparable Reconstruction Accuracy**: Achieves RMSE performance comparable to MissForest and other state-of-the-art methods across three missingness mechanisms (MCAR, MAR, MNAR)

✓ **Physical Plausibility**: Automatically respects physical bounds and constraints on astrophysical parameters, correcting unphysical imputations

✓ **Calibrated Uncertainty Estimates**: Provides ensemble-based uncertainty quantification across multiple imputation draws, enabling principled downstream analysis

✓ **Structure Preservation**: Maintains the covariance structure and principal component geometry of the original dataset—critical for population-level analyses

## The Problem: Missing Data in Stellar Host Surveys

Astronomical surveys of exoplanet host stars suffer from systematic biases and missing data:

- **Incomplete observations**: Not all stellar parameters are measured for every star
- **Observational bias**: Measurements are often biased toward brighter, closer, or more massive stars
- **Sparse coverage**: Some parameters (e.g., stellar age) are extremely sparse across large surveys
- **Population-level analysis**: Drawing conclusions about host star properties requires understanding both the observed data and the missing patterns

Traditional imputation methods fail because:

- Simple methods (mean/median) destroy covariance structure
- Deterministic methods provide no uncertainty quantification
- Standard approaches don't respect physical constraints on stellar parameters

## Methodology: SWRF-Impute

SWRF-Impute combines multiple statistical techniques into a coherent framework:

### 1. **Initialization**

Missing values are initialized from a clean, complete initial distribution (`X_init`), preserving the distributional properties of observed stellar parameters.

### 2. **Stochastic Gibbs Sampling**

The algorithm iteratively estimates missing values through multiple Gibbs sampling iterations:

- **Iteration phase**: Train random forests for each feature using observed values, weighted by the completeness of each observation
- **Draw phase**: Generate multiple stochastic draws from a truncated normal distribution centered on forest predictions

### 3. **Weighted Random Forests**

Unlike standard imputation:

- Each training sample is weighted by its **observation completeness ratio** — rows with more observed values receive higher weight
- This prevents sparse, heavily-imputed rows from dominating the regression model
- Random forests capture complex, non-linear relationships between stellar parameters

### 4. **Physical Bounds**

Predictions are constrained to physically plausible ranges:

- Each stellar parameter is bounded by the range observed in both the original and initialization datasets
- Drawn values are sampled from truncated normal distributions respecting these bounds
- Prevents physically impossible stellar parameters (e.g., negative mass, negative or >13byr age, etc.)

### 5. **Uncertainty Quantification**

The algorithm returns **multiple posterior samples** rather than point estimates:

- Stochastic draws across iterations capture posterior uncertainty
- Ensemble averaging produces both point estimates and credible intervals
- Uncertainty encompasses both parameter uncertainty and observational bias

## Validation Methodology

We validate SWRF-Impute across three realistic missingness mechanisms:

### **MCAR Validation**

- Randomly removes data uniformly across all parameters
- Tests baseline imputation capability without systematic bias
- Simulates unbiased missing data scenarios

### **MAR Validation**  

- Missingness depends on observed values
- A "trigger" feature determines which rows have missing data
- Trigger feature remains fully observed
- Tests if the method handles data-dependent missingness

### **MNAR Validation**

- Missingness depends on **unobserved values**
- Missing Based on Unobserved Variables (MBUV) strategy simulates hidden confounding
- Only one random feature per affected row is fully observed
- Mimics observational bias common in real surveys (e.g., measurement difficulty correlates with stellar type)

### **Evaluation Metrics**

1. **Reconstruction Accuracy**: Root Mean Squared Error (RMSE) on imputed values compared to held-out ground truth
2. **Structure Preservation**: Log L2 between eigenvalue spectra, measuring Principal Component geometry preservation
3. **Robustness**: Performance tracked across missingness levels from 10% to 90%

## Why This Matters for Stellar Host Analysis

Stellar host surveys are **population-level studies** where understanding both individual uncertainty and population structure is critical:

1. **Biased Datasets**: Most stellar host surveys are biased toward bright, nearby stars. SWRF-Impute's uncertainty estimates account for observational bias embedded in the missing data patterns.

2. **Population Inference**: Studies of exoplanet host properties require unbiased population statistics. Preserving the covariance structure ensures that relationships between stellar parameters (e.g., mass-radius relations) are maintained.

3. **Downstream Analysis**: MNAR validation demonstrates that SWRF-Impute handles the most realistic missing data mechanism, where unobserved confounders drive missingness—exactly the scenario in practice.

4. **Calibrated Confidence**: Ensemble uncertainty estimates enable principled statistical tests and confidence intervals for population properties.

## Data

The analysis uses stellar and exoplanet host data from:

- **Gaia Data Release 3**: Astrometric and stellar parameters (luminosity, radius, mass, age, metallicity, surface gravity, effective temperature)
- **NASA Exoplanet Archive**: Exoplanet host metadata
