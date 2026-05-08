# Meta-SHAP: Explainable AI and Meta-Regression Framework for RCT Subgroup Analysis

Meta-SHAP is an analytical pipeline designed to deconstruct treatment effect heterogeneity from aggregate-level randomized controlled trial (RCT) data. 

By integrating multilevel meta-regression (R) with an inverse-variance weighted XGBoost-SHAP architecture (Python), this framework identifies and statistically validates the core clinical predictors driving efficacy differences. While originally developed for oncology, the pipeline is entirely disease-agnostic and applicable to any meta-analytic dataset with a subgroup structure.

## Repository Structure

- `Perioperative_Subgroup_Analysis.R`: R script for multilevel meta-regression to estimate adjusted hazard ratios.
- `XGBoost_SHAP_Pipeline.py`: Main Python script for model training, SHAP extraction, and visualization.
- `validation.py`: Helper module for strict permutation testing (dependency for Python pipeline).
- `data.csv`: Standardized input dataset (Primary Outcomes).

## Data Format

Both R and Python pipelines expect a CSV file named `data.csv` with a consistent 6-column structure. 

### Data Dictionary
| Column | Description | Example |
| :--- | :--- | :--- |
| `Trial` | Clinical trial identifier | *CheckMate-816* |
| `Category` | Feature classification | *pd_l1_expression* |
| `Subgroup` | Specific clinical subgroup | *PD-L1 < 1%* |
| `HR` | Observed Hazard Ratio (or equivalent) | *0.65* |
| `Lower_CI` | 95% CI Lower Bound | *0.42* |
| `Upper_CI` | 95% CI Upper Bound | *0.91* |

### Reproducing Paper Results
To support the findings in our study, we provide two specific datasets:
1. **`data.csv`**: Contains survival outcomes (e.g., EFS) used for the primary analysis.
2. **`data_pcr_mpr.csv`**: Contains pathologic response data (pCR and mR). To analyze this file, rename it to `data.csv` before running the scripts, or modify the `input_file` variable in the code.

## Usage

### 1. Multilevel Meta-Regression (R Pipeline)
```bash
Rscript Perioperative_Subgroup_Analysis.R
```
### 2. Machine Learning & Explainable AI (Python Pipeline)
```bash
python XGBoost_SHAP_Pipeline.py
```
## Outputs
Generated files are saved in the output/ directory:

Visuals (PDF): Forest plots, SHAP summary, LOTO sensitivity heatmap, and permutation test panels.

Data (CSV): Adjusted HR summaries, raw SHAP values, and statistical validation metrics.

## License
MIT License
