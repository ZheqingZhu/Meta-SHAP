# ==============================================================================
# XGBoost-SHAP Pipeline for Perioperative Immunotherapy Efficacy
#
# METHODOLOGICAL NOTE: DECOUPLED VISUAL AND STATISTICAL VALIDATION STRATEGY
# To ensure the highest level of methodological integrity, this pipeline separates
# clinical interpretation from algorithmic validation:
#
# 1. Clinical Interpretation (Figure 2): Post-hoc SHAP consolidation is applied
#    exclusively to the main summary plot. This resolves perfect multicollinearity
#    (e.g., Squamous vs. Non-squamous) to align with biological phenotypes.
# 2. Algorithmic Validation (Figures S1, S2, S4): All rigorous statistical
#    evaluations (LOTO sensitivity, Permutation testing, Bootstrap CIs) are
#    intentionally executed on the RAW, UNCONSOLIDATED algorithmic outputs.
#    This strictly evaluates the native stability of the XGBoost engine without
#    introducing subjective mathematical interventions during resampling, serving
#    as a highly conservative evaluation of model robustness.
# ==============================================================================

import os
import warnings
import random
import copy
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats

# Ensure validation.py is in the same directory (Retained for native statistical validation)
try:
    from validation import run_permutation_test
except ImportError:
    raise ImportError("Please ensure 'validation.py' is located in the same directory.")

# ==============================================================================
# Global Configuration for Reproducibility
# ==============================================================================
GLOBAL_SEED = 42
os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

npg_blue = "#4DBBD5"
npg_red = "#E64B35"
npg_light = "#EEEEEE"
nature_cmap = mcolors.LinearSegmentedColormap.from_list("npg_shap", [npg_blue, npg_light, npg_red])

# ==============================================================================
# 1. Path Setup & Data Loading
# ==============================================================================
base_dir = "./"
data_path = os.path.join(base_dir, "data.csv")
out_dir = os.path.join(base_dir, "output")

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print("Loading and preprocessing data...")
df = pd.read_csv(data_path)

df = df[df['Category'] != 'Overall'].copy()
df = df[(df['HR'] > 0) & (df['Lower_CI'] > 0) & (df['Upper_CI'] > 0)].copy()

df['log_HR'] = np.log(df['HR'])
df['SE'] = (np.log(df['Upper_CI']) - np.log(df['Lower_CI'])) / (2 * 1.96)
df['weight'] = 1 / (df['SE'] ** 2)
df = df[(df['SE'] > 0) & (df['weight'] != np.inf)].dropna(subset=['HR', 'Lower_CI', 'Upper_CI'])

# ==============================================================================
# 2. Feature Engineering (Preserving pure structure for native validation)
# ==============================================================================
features = ['Trial', 'Category', 'Subgroup']
X_raw = df[features].copy()
y = df['log_HR']
weights = df['weight']

X = pd.get_dummies(X_raw, columns=features)
X = X.astype(float)

X.columns = X.columns.str.replace('<', 'lt_', regex=False)
X.columns = X.columns.str.replace('>', 'gt_', regex=False)
X.columns = X.columns.str.replace('[', '_', regex=False)
X.columns = X.columns.str.replace(']', '_', regex=False)

# ==============================================================================
# 3. Model Training
# ==============================================================================
print("Training inverse-variance weighted XGBoost model...")
best_params = {
    'n_estimators': 300,
    'max_depth': 3,
    'learning_rate': 0.05,
    'random_state': GLOBAL_SEED,
    'subsample': 0.8,
    'objective': 'reg:squarederror'
}
model = xgb.XGBRegressor(**best_params)
model.fit(X, y, sample_weight=weights)

# ==============================================================================
# 4. SHAP Value Extraction (Raw values for stats)
# ==============================================================================
print("Calculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values_raw = explainer(X)

MEDICAL_RENAME_DICT = {
    "Subgroup_Female": "Female sex", "Subgroup_Male": "Male sex",
    "Subgroup_Squamous": "Squamous histology", "Subgroup_Nonsquamous": "Non-squamous histology",
    "Subgroup_Current smoker": "Smoking: current", "Subgroup_Former smoker": "Smoking: former",
    "Subgroup_Never smoked": "Smoking: never", "Subgroup_Current or former smoker": "Smoking: current/former",
    "Subgroup_Never or former smoker": "Smoking: never/former",
    "Subgroup_lt_1%": "PD-L1 < 1%", "Subgroup_1-49%": "PD-L1 1-49%",
    "Subgroup_gt_=50%": "PD-L1 ≥ 50%", "Subgroup_gt_=1%": "PD-L1 ≥ 1%",
    "Subgroup_Not evaluable": "PD-L1 not evaluable",
    "Subgroup_Stage II": "Clinical stage II", "Subgroup_Stage III": "Clinical stage III",
    "Subgroup_IB or II": "Clinical stage IB/II", "Subgroup_IIA": "Clinical stage IIA",
    "Subgroup_IIB": "Clinical stage IIB", "Subgroup_IIIA": "Clinical stage IIIA",
    "Subgroup_IIIB": "Clinical stage IIIB", "Subgroup_III N2": "Clinical stage III (N2)",
    "Subgroup_III non-N2": "Clinical stage III (non-N2)",
    "Subgroup_N0": "Nodal status: N0", "Subgroup_N1": "Nodal status: N1",
    "Subgroup_N2": "Nodal status: N2", "Subgroup_N2 Single-station": "Nodal status: N2 single",
    "Subgroup_N2 Multi-station": "Nodal status: N2 multi",
    "Subgroup_T1-3": "Tumor status: T1-T3", "Subgroup_T4": "Tumor status: T4",
    "Subgroup_0": "ECOG PS 0", "Subgroup_1": "ECOG PS 1",
    "Subgroup_Asian": "Race: Asian", "Subgroup_Non-Asian": "Race: non-Asian",
    "Subgroup_White": "Race: White", "Subgroup_Non-White": "Race: non-White",
    "Subgroup_Asia": "Region: Asia", "Subgroup_East Asia": "Region: East Asia",
    "Subgroup_Non-East Asia": "Region: non-East Asia", "Subgroup_Europe": "Region: Europe",
    "Subgroup_North America": "Region: North America", "Subgroup_South America": "Region: South America",
    "Subgroup_lt_65 yr": "Age < 65", "Subgroup_gt_=65 yr": "Age ≥ 65",
    "Subgroup_Cisplatin": "Chemotherapy: cisplatin", "Subgroup_Carboplatin": "Chemotherapy: carboplatin",
    "Subgroup_Positive": "ctDNA/biomarker positive", "Subgroup_Negative": "ctDNA/biomarker negative",
    "Subgroup_lt_=25": "TMB ≤ 25", "Subgroup_gt_25": "TMB > 25",
    "Subgroup_lt_12.3": "TMB < 12.3", "Subgroup_gt_=12.3": "TMB ≥ 12.3",
    "Subgroup_No": "No", "Subgroup_Yes": "Yes"
}


def get_clean_name(col):
    if col in MEDICAL_RENAME_DICT: return MEDICAL_RENAME_DICT[col]
    auto_clean = col.replace('Subgroup_', '').replace('Trial_', 'Trial: ').replace('Category_', 'Category: ')
    auto_clean = auto_clean.replace('gt_=', '≥ ').replace('lt_=', '≤ ').replace('lt_', '< ').replace('gt_', '> ')
    return auto_clean.capitalize()


clean_feature_names = [get_clean_name(col) for col in X.columns]
if hasattr(shap_values_raw, 'feature_names'):
    shap_values_raw.feature_names = clean_feature_names

# ==============================================================================
# 5. Figure 2: SHAP Summary (Visual Consolidation ONLY via Deep Copy)
# ==============================================================================
print("Generating Figure 2 and data...")

# Create a deepcopy specifically for Figure 2 so RAW values remain untouched for stats.
shap_values_fig2 = copy.deepcopy(shap_values_raw)

if 'Non-squamous histology' in clean_feature_names and 'Squamous histology' in clean_feature_names:
    idx_ns = clean_feature_names.index('Non-squamous histology')
    idx_sq = clean_feature_names.index('Squamous histology')
    shap_values_fig2.values[:, idx_sq] += shap_values_fig2.values[:, idx_ns]
    shap_values_fig2.values[:, idx_ns] = 0

for feat in clean_feature_names:
    if feat.startswith('Category:') and 'nodal_status' not in feat.lower():
        idx_r = clean_feature_names.index(feat)
        shap_values_fig2.values[:, idx_r] = 0

X_display = X.copy()
X_display.columns = clean_feature_names

plt.figure(figsize=(10, 5))
shap.summary_plot(shap_values_fig2, X_display, show=False, max_display=15, cmap=nature_cmap, plot_size=(10, 5.5))
plt.xlabel("SHAP value (impact on immunotherapy efficacy variance)", fontsize=11)
plt.savefig(os.path.join(out_dir, "Figure2_SHAP_Summary.pdf"), format='pdf', bbox_inches='tight')

mean_abs_shap_fig2 = np.abs(shap_values_fig2.values).mean(axis=0)
df_fig2_summary = pd.DataFrame({'Feature': clean_feature_names, 'Mean_Abs_SHAP': mean_abs_shap_fig2}).sort_values(
    by='Mean_Abs_SHAP', ascending=False)
df_fig2_summary.to_csv(os.path.join(out_dir, "Figure2_Data_MeanAbs_SHAP.csv"), index=False)
pd.DataFrame(shap_values_fig2.values, columns=clean_feature_names).to_csv(
    os.path.join(out_dir, "Figure2_Data_Raw_SHAP.csv"), index=False)

# ==============================================================================
# 6. Figure S1: LOTO Sensitivity Heatmap (NATIVE ALGORITHMIC VALIDATION)
# ==============================================================================
print("Generating Figure S1 (LOTO Sensitivity) and data...")
trial_cols = [col for col in X.columns if 'Trial' in col]
all_rankings = {}

for trial_col in trial_cols:
    trial_display_name = trial_col.replace('Trial_', '').replace('Trial: ', '')
    mask = X[trial_col] == 0
    X_subset = X[mask]
    if len(X_subset) == 0: continue

    model_sens = xgb.XGBRegressor(**best_params).fit(X_subset, y[mask], sample_weight=weights[mask])
    shap_vals_sens = shap.TreeExplainer(model_sens)(X_subset)

    # Using RAW mean absolute SHAP (No post-hoc consolidation)
    mean_abs_sens = np.abs(shap_vals_sens.values).mean(axis=0)

    clinical_mask = [not ('Trial' in col) for col in X_subset.columns]
    raw_clinical_names = X_subset.columns[clinical_mask]
    clinical_shap = mean_abs_sens[clinical_mask]

    ranks = len(clinical_shap) - np.argsort(np.argsort(clinical_shap))
    beautified_names = [get_clean_name(name) for name in raw_clinical_names]
    all_rankings[f"Excl. {trial_display_name}"] = dict(zip(beautified_names, ranks))

asian_trials = ['Trial_Neotorch', 'Trial_RATIONALE_315']
valid_asian_trials = [t for t in asian_trials if t in X.columns]
if valid_asian_trials:
    condition = np.zeros(len(X), dtype=bool)
    for t in valid_asian_trials: condition = condition | (X[t] == 1)
    mask_exclude_asia = ~condition
    X_subset_global = X[mask_exclude_asia]
    if len(X_subset_global) > 0:
        model_global = xgb.XGBRegressor(**best_params).fit(X_subset_global, y[mask_exclude_asia],
                                                           sample_weight=weights[mask_exclude_asia])
        shap_vals_global = shap.TreeExplainer(model_global)(X_subset_global)
        clinical_shap_global = np.abs(shap_vals_global.values).mean(axis=0)[clinical_mask]
        ranks_global = len(clinical_shap_global) - np.argsort(np.argsort(clinical_shap_global))
        all_rankings["Excl. Asian-only cohorts"] = dict(zip(beautified_names, ranks_global))

rank_df = pd.DataFrame(all_rankings)
top_features_sens = rank_df.mean(axis=1).sort_values().head(12).index
rank_df_top_sens = rank_df.loc[top_features_sens]

plt.figure(figsize=(11, 5.5))
sns.heatmap(rank_df_top_sens, annot=True, cmap=sns.light_palette(npg_red, reverse=True, as_cmap=True), linewidths=1.5,
            linecolor='white', fmt="g", vmin=1, vmax=10)
plt.ylabel("Standardized clinical signatures", fontsize=11, fontweight='bold')
plt.xlabel("Sensitivity scenarios (excluded cohorts)", fontsize=11, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.savefig(os.path.join(out_dir, "FigureS1_Sensitivity_Heatmap.pdf"), format='pdf', bbox_inches='tight')

rank_df_top_sens.to_csv(os.path.join(out_dir, "FigureS1_Data_Sensitivity_Ranks.csv"))

# ==============================================================================
# 7. Figure S2: Permutation Test (NATIVE ALGORITHMIC VALIDATION)
# ==============================================================================
print("Executing Permutation Test...")
target_features = ['Subgroup_gt_=50%', 'Subgroup_lt_1%', 'Subgroup_T4']

true_importance, permuted_importances, p_vals = run_permutation_test(X=X, y=y, params=best_params, n_permutations=1000)

figS2_summary_data = []
figS2_perm_dist_data = {}
fig = plt.figure(figsize=(12, 3.5))
gs = gridspec.GridSpec(1, 3, wspace=0.3)
color_null = "#B0B0B0"
color_obs = "#E64B35"

for idx, feature in enumerate(target_features):
    if feature not in X.columns: continue
    ax = fig.add_subplot(gs[0, idx])
    feat_idx = list(X.columns).index(feature)
    p_val = p_vals[feature]
    clean_name = get_clean_name(feature)
    obs_val = true_importance[feat_idx]

    figS2_summary_data.append({'Feature': clean_name, 'Observed_SHAP': obs_val, 'P_value': p_val})
    figS2_perm_dist_data[clean_name] = permuted_importances[:, feat_idx]

    ax.hist(permuted_importances[:, feat_idx], bins=20, alpha=0.8, color=color_null, edgecolor='white', linewidth=0.5,
            label='Null distribution')
    ax.axvline(obs_val, color=color_obs, linestyle='--', lw=2.5, label='Observed')
    ax.spines['top'].set_visible(False);
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2);
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(width=1.2, labelsize=10)
    ax.set_xlabel('Mean absolute SHAP value', fontsize=10)
    if idx == 0: ax.set_ylabel('Frequency', fontsize=10)
    annot_text = f"{clean_name}\nP = {p_val:.3f}"
    ax.annotate(annot_text, xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top', fontweight='bold',
                fontsize=11, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    if idx == 2: ax.legend(frameon=False, loc='center right', fontsize=10)

plt.savefig(os.path.join(out_dir, "FigureS2_Permutation_Panel_1x3.pdf"), format='pdf', bbox_inches='tight')
pd.DataFrame(figS2_summary_data).to_csv(os.path.join(out_dir, "FigureS2_Data_Permutation_Summary.csv"), index=False)
pd.DataFrame(figS2_perm_dist_data).to_csv(os.path.join(out_dir, "FigureS2_Data_Permutation_Distributions.csv"),
                                          index=False)

# ==============================================================================
# 8. Bootstrap Confidence Intervals (NATIVE ALGORITHMIC VALIDATION)
# ==============================================================================
print("Executing Bootstrap for Confidence Intervals...")
n_bootstraps = 1000
boot_importances = []
for i in range(n_bootstraps):
    X_boot, y_boot, w_boot = resample(X, y, weights, random_state=i)
    model_boot = xgb.XGBRegressor(**best_params).fit(X_boot, y_boot, sample_weight=w_boot)
    # Deliberately using RAW unconsolidated values to compute maximum variance bounds
    boot_importances.append(np.abs(shap.TreeExplainer(model_boot)(X_boot).values).mean(axis=0))

boot_importances = np.array(boot_importances)
lower_ci = np.percentile(boot_importances, 2.5, axis=0)
upper_ci = np.percentile(boot_importances, 97.5, axis=0)

# Use true_importance (RAW, un-consolidated) to perfectly align with published validation.py
mean_shap_raw = true_importance

# ==============================================================================
# 9. Figure S4: PD-L1 Gradient Plot
# ==============================================================================
print("Generating Figure S4 (PD-L1 Gradient)...")
pdl1_features = ['Subgroup_lt_1%', 'Subgroup_1-49%', 'Subgroup_gt_=50%']
pdl1_names = ['PD-L1 < 1%', 'PD-L1 1-49%', 'PD-L1 ≥ 50%']
pdl1_means = [mean_shap_raw[list(X.columns).index(f)] if f in X.columns else 0 for f in pdl1_features]
pdl1_lower = [lower_ci[list(X.columns).index(f)] if f in X.columns else 0 for f in pdl1_features]
pdl1_upper = [upper_ci[list(X.columns).index(f)] if f in X.columns else 0 for f in pdl1_features]
pdl1_errs = [(u - l) / 2 for u, l in zip(pdl1_upper, pdl1_lower)]

plt.figure(figsize=(6, 5))
plt.errorbar(pdl1_names, pdl1_means, yerr=pdl1_errs, fmt='-o', color=npg_red, ecolor=npg_light, elinewidth=3, capsize=5,
             markersize=10, linewidth=2)
plt.ylabel("Mean absolute SHAP value\n(impact magnitude)", fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.gca().spines['top'].set_visible(False);
plt.gca().spines['right'].set_visible(False)
plt.savefig(os.path.join(out_dir, "FigureS4_PDL1_Gradient.pdf"), format='pdf', bbox_inches='tight')

df_figS4 = pd.DataFrame({
    'Subgroup': pdl1_names, 'Mean_Abs_SHAP': pdl1_means,
    'Lower_95CI': pdl1_lower, 'Upper_95CI': pdl1_upper
})
df_figS4.to_csv(os.path.join(out_dir, "FigureS4_Data_PDL1_Gradient.csv"), index=False)

# ==============================================================================
# 10. Figure S5: Model Calibration Plot
# ==============================================================================
print("Generating Figure S5 (Calibration)...")
y_pred = model.predict(X)
r2, rmse = r2_score(y, y_pred), np.sqrt(mean_squared_error(y, y_pred))

# Unweighted regression preserved for Unbiased Macro-Averaged Calibration check
slope, intercept, _, _, _ = stats.linregress(y, y_pred)

plt.figure(figsize=(6, 6))
plt.scatter(y, y_pred, alpha=0.7, color=npg_blue, s=60, edgecolor='white', linewidth=0.5)
min_val, max_val = min(y.min(), y_pred.min()) - 0.1, max(y.max(), y_pred.max()) + 0.1
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal alignment ($y=x$)')
plt.plot([min_val, max_val], intercept + slope * np.array([min_val, max_val]), color=npg_red, lw=2,
         label=f'Model fit (slope={slope:.2f})')
plt.xlabel("Observed $\\ln(\\mathrm{HR})$ from RCTs", fontsize=11)
plt.ylabel("In-Sample XGBoost fitted $\\ln(\\mathrm{HR})$", fontsize=11)
plt.text(0.05, 0.95, f"In-Sample $R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nUnweighted Slope = {slope:.2f}",
         transform=plt.gca().transAxes, va='top',
         bbox=dict(boxstyle='round', facecolor=npg_light, alpha=0.8, edgecolor='none'))
plt.legend(frameon=False, loc='lower right')
plt.gca().spines['top'].set_visible(False);
plt.gca().spines['right'].set_visible(False)
plt.savefig(os.path.join(out_dir, "FigureS5_Calibration.pdf"), format='pdf', bbox_inches='tight')

pd.DataFrame({'Observed_log_HR': y, 'Predicted_log_HR': y_pred}).to_csv(
    os.path.join(out_dir, "FigureS5_Data_Calibration.csv"), index=False)

# ==============================================================================
# 11. Baseline Model Benchmarking (Explanatory Paradigm)
# ==============================================================================
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

print("Benchmarking baseline models (In-Sample Explained Variance)...")
model_en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=GLOBAL_SEED, max_iter=10000).fit(X, y,
                                                                                              sample_weight=weights)
y_pred_en = model_en.predict(X)
model_rf = RandomForestRegressor(n_estimators=300, max_depth=3, random_state=GLOBAL_SEED, n_jobs=-1).fit(X, y,
                                                                                                         sample_weight=weights)
y_pred_rf = model_rf.predict(X)


def get_metrics(y_true, y_p):
    return np.sqrt(mean_squared_error(y_true, y_p)), r2_score(y_true, y_p), stats.linregress(y_true, y_p)[0]


metrics_en = get_metrics(y, y_pred_en)
metrics_rf = get_metrics(y, y_pred_rf)
metrics_xgb = get_metrics(y, y_pred)

print("\n===================================================================")
print(" IN-SAMPLE EXPLAINED VARIANCE BENCHMARKING (Table 2) ")
print(" Note: Metrics represent in-sample goodness-of-fit. They demonstrate ")
print(" the model's explanatory capacity to capture structural heterogeneity. ")
print(" Evaluation metrics are deliberately UNWEIGHTED to reflect uniform baselines.")
print("===================================================================")
print(f"{'Model Architecture':<35} | {'RMSE':<6} | {'R^2':<6} | {'Calib. Slope':<12}")
print("-" * 70)
print(f"{'Inverse-Variance Elastic Net':<35} | {metrics_en[0]:.3f}  | {metrics_en[1]:.3f}  | {metrics_en[2]:.3f}")
print(f"{'Inverse-Variance Random Forest':<35} | {metrics_rf[0]:.3f}  | {metrics_rf[1]:.3f}  | {metrics_rf[2]:.3f}")
print(f"{'Inverse-Variance XGBoost':<35} | {metrics_xgb[0]:.3f}  | {metrics_xgb[1]:.3f}  | {metrics_xgb[2]:.3f}")
print("===================================================================\n")

if 'Subgroup_T4' in X.columns:
    t4_idx = list(X.columns).index('Subgroup_T4')
    print(f"T4 95% CI: {lower_ci[t4_idx]:.4f} - {upper_ci[t4_idx]:.4f}")
