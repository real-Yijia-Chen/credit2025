# Credit2025: Measuring Credit Environment & Regional Economic Growth

This repository contains Python scripts to replicate and extend an empirical analysis of China’s regional credit environment and its impact on economic growth. The project cleans and merges multiple data sources, performs exploratory data analysis, runs panel regressions, and conducts robustness checks.

## Project Overview

- **Objective:** Quantify how different measures of the regional credit environment (expert weighting, PCA, equal-weight) affect per-capita GDP growth across Chinese provinces.
- **Workflow:**  
  1. Data cleaning & merging  
  2. Exploratory visualization  
  3. Panel regression analysis  
  4. Non-parametric robustness tests  

## File Descriptions

| File                                | Description                                                                                       |
| :--------------------------------- | :------------------------------------------------------------------------------------------------ |
| `clean_data.py`                     | Load raw Excel inputs, fill missing values, merge mediator variables (entrepreneurship, innovation, consumption, trade, FDI) into a single dataset. |
| `exploratory_data_analysis.py`      | Produce scatter plots & line charts showing credit measures vs. regional per-capita GDP trends.    |
| `regression_analysis.py`            | Estimate panel regressions (FDI mediating effect, social credit dimensions) and export coefficient tables to Excel. |
| `robustness_check.py`               | Perform Friedman & Nemenyi tests on city-level credit rankings and output test statistics.        |

## Key Features

- **Multiple Credit Measures:** Expert weighting, principal component analysis (PCA), and equal-weight indexing.  
- **Mediator Analysis:** Tests how FDI and social credit sub-indices mediate credit’s impact on GDP.  
- **Panel Regression:** Controls for fixed effects and clustered standard errors.  
- **Robustness Tests:** Non-parametric Friedman & Nemenyi procedures on city rankings.  
- **Modular Design:** Each script can be run independently or in sequence for reproducibility.

## Requirements

- Python 3.8+  
- pandas  
- numpy  
- matplotlib  
- scipy  
- statsmodels  

```bash
pip install pandas numpy matplotlib scipy statsmodels
