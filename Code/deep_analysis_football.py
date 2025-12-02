"""
Script: deep_analysis_football.py
Author: Mostafa Shams
Description:
    This script performs the "Deep Dive" forensic analysis required for top-tier
    peer review.

    SCIENTIFIC UPDATES:
    1. Explainable AI (SHAP): Quantify feature importance (Fundamentals vs Odds).
    2. Alpha Decay: Analyze year-over-year ROI degradation.
    3. Kelly Criterion: Test probability calibration via staking efficiency.
    4. Expected Calibration Error (ECE): Metric for model overconfidence.
    5. Concept Drift: KS-Test to detect structural market shifts (Early vs Late era).

    Input: master_feature_table.csv
    Output: Figures and report_deep_analysis.txt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import xgboost as xgb
import shap
from scipy.stats import ks_2samp
import warnings
import os

warnings.filterwarnings('ignore')

INPUT_FILE = "master_feature_table.csv"
OUTPUT_REPORT = "report_deep_analysis.txt"

# --- CONFIGURATION ---
KELLY_FRACTION = 0.05
INITIAL_BANKROLL = 10000
FLAT_STAKE = 100

# --- VISUAL STYLE SETUP ---
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.figsize': (16, 9),
    'figure.facecolor': '#f8f9fa',
    'axes.facecolor': '#f8f9fa',
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'grid.linestyle': '--',
    'grid.alpha': 0.6,
})

# --- NEW METRIC FUNCTIONS ---

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculates ECE: The weighted average difference between confidence and accuracy.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_conf = np.mean(y_prob[mask])
            bin_acc = np.mean(y_true[mask])
            bin_count = np.sum(mask)
            ece += (bin_count / total_samples) * np.abs(bin_acc - bin_conf)
            
    return ece

def analyze_concept_drift(df):
    """
    Performs Kolmogorov-Smirnov Test to detect Shift in Market Efficiency.
    Compares Early Era (2006-2012) vs Late Era (2015-2021).
    """
    print("  -> Analyzing Concept Drift (Market Adaptation)...")
    
    # Define Eras
    early_era = df[(df['Season'] >= 2006) & (df['Season'] <= 2012)]
    late_era = df[(df['Season'] >= 2015) & (df['Season'] <= 2021)]
    
    if len(early_era) == 0 or len(late_era) == 0:
        return None

    # 1. Prediction Drift: Are we predicting differently?
    ks_stat_prob, p_value_prob = ks_2samp(early_era['Prob_H'], late_era['Prob_H'])
    
    # 2. Residual Drift: Are the errors distributed differently?
    # Brier Score per match for Home Prediction
    early_brier = (early_era['Prob_H'] - (early_era['FTR'] == 'H').astype(int))**2
    late_brier = (late_era['Prob_H'] - (late_era['FTR'] == 'H').astype(int))**2
    
    ks_stat_err, p_value_err = ks_2samp(early_brier, late_brier)
    
    return {
        'prob_drift': (ks_stat_prob, p_value_prob),
        'error_drift': (ks_stat_err, p_value_err),
        'early_mean_err': early_brier.mean(),
        'late_mean_err': late_brier.mean()
    }

# --- EXISTING PIPELINE FUNCTIONS ---

def get_features():
    return [
        'HomeOdds', 'DrawOdds', 'AwayOdds',
        'Home_Avg_Points_L5', 'Home_Avg_GoalsConceded_L5',
        'Home_Avg_Shots_L5', 'Home_Avg_ShotsTarget_L5', 'Home_Avg_Corners_L5',
        'Away_Avg_Points_L5', 'Away_Avg_GoalsConceded_L5',
        'Away_Avg_Shots_L5', 'Away_Avg_ShotsTarget_L5', 'Away_Avg_Corners_L5'
    ]

def load_data():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} not found.")
    
    df = pd.read_csv(INPUT_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    features = get_features()
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df.dropna(subset=features + ['FTR'], inplace=True)
    df['Season'] = np.where(df['Date'].dt.month >= 8, df['Date'].dt.year, df['Date'].dt.year - 1)
    df['Target'] = df['FTR'].map({'H': 2, 'D': 1, 'A': 0})
    df.dropna(subset=['Target'], inplace=True)
    df['Target'] = df['Target'].astype(int)
    
    return df

def train_and_explain_shap(df, features):
    print("  -> Training XGBoost on full dataset for SHAP analysis...")
    X = df[features]; y = df['Target']
    model = xgb.XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss', 
                              use_label_encoder=False, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    print("  -> Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    shap_vals_home = shap_values[2] if isinstance(shap_values, list) else shap_values[:, :, 2]
    if shap_vals_home is None or len(shap_vals_home.shape) != 2:
        shap_vals_home = np.array(shap_values)
    
    fig = plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_vals_home, X, show=False, plot_type="bar", color='#0066CC')
    ax = plt.gca()
    ax.set_title("SHAP Feature Importance (Target: Home Win)", fontsize=20, pad=20)
    plt.tight_layout()
    plt.savefig("fig_shap_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    vals = np.abs(shap_vals_home).mean(0)
    feature_importance = pd.DataFrame(list(zip(features, vals)), columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    return feature_importance

def run_walk_forward_inference(df, features):
    print("  -> Running Walk-Forward Inference...")
    seasons = sorted(df['Season'].unique())
    start_idx = 5
    predictions = []
    
    for i in range(start_idx, len(seasons)):
        test_season = seasons[i]
        train = df[df['Season'] < test_season]
        test = df[df['Season'] == test_season]
        if len(test) == 0: continue
        
        model = xgb.XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss', 
                                  use_label_encoder=False, random_state=42, n_jobs=-1)
        model.fit(train[features], train['Target'])
        probs = model.predict_proba(test[features])
        preds = model.predict(test[features])
        
        temp_df = test.copy()
        temp_df['Prob_A'] = probs[:, 0]; temp_df['Prob_D'] = probs[:, 1]; temp_df['Prob_H'] = probs[:, 2]
        temp_df['Pred_Code'] = preds
        temp_df['Pred_Str'] = temp_df['Pred_Code'].map({0:'A', 1:'D', 2:'H'})
        predictions.append(temp_df)
        
    return pd.concat(predictions)

def analyze_alpha_decay(df):
    print("  -> Analyzing Alpha Decay...")
    df['Won'] = df['Pred_Str'] == df['FTR']
    conditions = [df['Pred_Str'] == 'H', df['Pred_Str'] == 'D', df['Pred_Str'] == 'A']
    choices = [df['HomeOdds'], df['DrawOdds'], df['AwayOdds']]
    df['BetOdds'] = np.select(conditions, choices, default=1.0)
    df['Flat_PnL'] = np.where(df['Won'], FLAT_STAKE * (df['BetOdds'] - 1), -FLAT_STAKE)
    
    seasonal = df.groupby('Season').agg({'Flat_PnL': 'sum', 'Date': 'count'}).rename(columns={'Date': 'Num_Bets'})
    seasonal['ROI'] = (seasonal['Flat_PnL'] / (seasonal['Num_Bets'] * FLAT_STAKE)) * 100
    
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(seasonal.index, seasonal['ROI'], marker='o', color='#001019', linewidth=2.5, label='Actual ROI')
    z = np.polyfit(seasonal.index, seasonal['ROI'], 1)
    p = np.poly1d(z)
    ax.plot(seasonal.index, p(seasonal.index), color='#d62828', linestyle='--', linewidth=2, label=f'Trend (Slope: {z[0]:.2f})')
    ax.axhline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.8)
    ax.fill_between(seasonal.index, seasonal['ROI'], 0, where=(seasonal['ROI'] >= 0), color='#2ca02c', alpha=0.15)
    ax.fill_between(seasonal.index, seasonal['ROI'], 0, where=(seasonal['ROI'] < 0), color='#d62828', alpha=0.15)
    ax.set_title("Alpha Decay: Evolution of Model Edge", fontsize=20, pad=15)
    ax.set_ylabel("ROI (%)")
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig("fig_alpha_decay.png", dpi=300)
    plt.close()
    return seasonal

def analyze_kelly_criterion(df):
    print("  -> Simulating Kelly Criterion...")
    conditions_p = [df['Pred_Str'] == 'H', df['Pred_Str'] == 'D', df['Pred_Str'] == 'A']
    choices_p = [df['Prob_H'], df['Prob_D'], df['Prob_A']]
    df['Model_Conf'] = np.select(conditions_p, choices_p, default=0.0)
    
    b = df['BetOdds'] - 1
    b = np.where(b <= 0, 0.01, b)
    p = df['Model_Conf']
    q = 1 - p
    df['Kelly_Full'] = (b * p - q) / b
    df['Kelly_Full'] = df['Kelly_Full'].clip(lower=0, upper=0.5)
    df['Stake_Pct'] = df['Kelly_Full'] * KELLY_FRACTION
    
    bankroll_flat = [INITIAL_BANKROLL]; bankroll_kelly = [INITIAL_BANKROLL]
    curr_flat = INITIAL_BANKROLL; curr_kelly = INITIAL_BANKROLL
    
    for idx, row in df.iterrows():
        curr_flat += row['Flat_PnL']
        bankroll_flat.append(curr_flat)
        stake_k = min(curr_kelly * row['Stake_Pct'], curr_kelly)
        pnl_k = stake_k * (row['BetOdds'] - 1) if row['Won'] else -stake_k
        curr_kelly += pnl_k
        bankroll_kelly.append(curr_kelly)
        
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(bankroll_flat, label="Flat Stake", color='gray', alpha=0.8)
    ax.plot(bankroll_kelly, label=f"Fractional Kelly ({KELLY_FRACTION*100}%)", color='#2ca02c')
    ax.axhline(INITIAL_BANKROLL, color='#d62828', linestyle='--')
    ax.set_title("Capital Growth: Flat vs Kelly", fontsize=20)
    ax.legend()
    plt.tight_layout()
    plt.savefig("fig_kelly_vs_flat.png", dpi=300)
    plt.close()
    return curr_kelly, curr_flat

def main():
    print("----------------------------------------------------------------")
    print("Task:   Forensic Analysis (SHAP, Alpha Decay, Kelly, ECE, Drift)")
    print("----------------------------------------------------------------")
    
    df_raw = load_data()
    features = get_features()
    
    shap_df = train_and_explain_shap(df_raw, features)
    df_preds = run_walk_forward_inference(df_raw, features)
    seasonal_data = analyze_alpha_decay(df_preds)
    end_kelly, end_flat = analyze_kelly_criterion(df_preds)
    
    # --- NEW: ECE CALCULATION ---
    # ECE for Home Wins (Target Class = 2 -> mapped to H)
    # df_preds['FTR'] is H, D, A.
    y_true_home = (df_preds['FTR'] == 'H').astype(int)
    ece_score = expected_calibration_error(y_true_home, df_preds['Prob_H'])
    
    # --- NEW: DRIFT ANALYSIS ---
    drift_metrics = analyze_concept_drift(df_preds)
    
    with open(OUTPUT_REPORT, "w") as f:
        f.write("DEEP DIVE FORENSIC REPORT\n")
        f.write("=========================\n\n")
        
        f.write("1. FEATURE IMPORTANCE (SHAP VALUES)\n")
        f.write(shap_df.head(10).to_string(index=False))
        f.write("\n\n")
        
        f.write("2. ALPHA DECAY (MARKET EFFICIENCY EVOLUTION)\n")
        f.write(seasonal_data[['ROI']].to_string(float_format="{:.2f}".format))
        f.write("\n\n")
        
        f.write("3. KELLY CRITERION VALIDATION\n")
        f.write(f"   - Final Bankroll (Flat):  ${end_flat:,.2f}\n")
        f.write(f"   - Final Bankroll (Kelly): ${end_kelly:,.2f}\n")
        if end_kelly < end_flat:
            f.write("   -> Kelly UNDERPERFORMED. Evidence of Miscalibration.\n")

        f.write("\n4. PROBABILITY CALIBRATION (ECE)\n")
        f.write(f"   Expected Calibration Error: {ece_score:.4f}\n")
        f.write("   Interpretation: The average gap between Model Confidence and Reality.\n")
        f.write("   > 0.05 implies poor calibration (explains Kelly failure).\n")
        
        if drift_metrics:
            f.write("\n5. CONCEPT DRIFT (KS-TEST)\n")
            f.write("   Comparing Early Era (2006-2012) vs Late Era (2015-2021):\n")
            f.write(f"   - Error Distribution Shift P-Value: {drift_metrics['error_drift'][1]:.4e}\n")
            f.write(f"   - Early Era Mean Brier Error: {drift_metrics['early_mean_err']:.4f}\n")
            f.write(f"   - Late Era Mean Brier Error:  {drift_metrics['late_mean_err']:.4f}\n")
            
            if drift_metrics['error_drift'][1] < 0.05:
                f.write("   -> STATISTICALLY SIGNIFICANT DRIFT DETECTED.\n")
                f.write("      The market has fundamentally changed structure.\n")
            else:
                f.write("   -> No significant drift detected.\n")
        
    print(f"\nAnalysis Complete. Report saved to {OUTPUT_REPORT}.")

if __name__ == "__main__":
    main()
