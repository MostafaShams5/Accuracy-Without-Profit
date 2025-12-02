"""
Script: statistical_significance_test.py
Author: Mostafa Shams

Description:
    This script performs a high-resolution, peer-review standard statistical analysis.
    
    UPDATES FOR PEER REVIEW:
    1. Paired T-Test: Checks if ROI is statistically different from benchmarks.
    2. Diebold-Mariano Test: Checks if the Model's FORECASTS (Probabilities) are 
       statistically distinguishable from the Bookmaker's implied probabilities.
       (Standard metric for Economic Forecasting).

    Input:
        - master_feature_table.csv
        - results_*.csv (Model Predictions)

    Output:
        - final_statistical_report.txt
        - fig_cumulative_wealth.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import os
import glob
import warnings

warnings.filterwarnings('ignore')

STAKE = 100
MIN_BETS_PER_WEEK = 5  
MASTER_FILE = "master_feature_table.csv"
BIG_SIX = ['Arsenal', 'Chelsea', 'Liverpool', 'Man United', 'Man City', 'Tottenham']

# --- NEW: DIEBOLD-MARIANO TEST ---
# --- REPLACE THIS FUNCTION IN statistical_significance_test.py ---

def diebold_mariano_test(y_true, y_pred_1, y_pred_2, h=1, criterion='brier'):
    """
    Performs the Diebold-Mariano test to compare forecast accuracy.
    H0: The two models have the same accuracy.
    """
    # Ensure inputs are numpy arrays of floats
    y_true = np.array(y_true, dtype=float)
    y_pred_1 = np.array(y_pred_1, dtype=float)
    y_pred_2 = np.array(y_pred_2, dtype=float)
    
    T = len(y_true)
    
    # Loss differential (Brier/Squared Error)
    e1 = (y_pred_1 - y_true)**2
    e2 = (y_pred_2 - y_true)**2
    d = e1 - e2
    mean_d = np.mean(d)
    
    # Autocovariance
    def autocovariance(d, k):
        # FIX: Handle k=0 case explicitly to avoid empty slice error
        if k == 0:
            return np.sum((d - mean_d) * (d - mean_d)) / T
        else:
            return np.sum((d[k:] - mean_d) * (d[:-k] - mean_d)) / T

    gamma = [autocovariance(d, k) for k in range(h)]
    var_d = gamma[0] + 2 * sum(gamma[1:])
    
    # Statistic
    if var_d <= 0: return 0.0, 1.0 # Handle edge case
    
    dm_stat = mean_d / np.sqrt(var_d / T)
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value

def get_weekly_roi_series(df, prediction_col, strategy_name):
    df_calc = df.copy()
    df_calc = df_calc.dropna(subset=[prediction_col])
    
    if 'FTR' in df_calc.columns: target_col = 'FTR'
    elif 'Actual' in df_calc.columns: target_col = 'Actual'
    else: return pd.DataFrame()

    req_cols = ['HomeOdds', 'DrawOdds', 'AwayOdds']
    if not all(col in df_calc.columns for col in req_cols): return pd.DataFrame()

    conditions = [
        df_calc[prediction_col] == 'H',
        df_calc[prediction_col] == 'D',
        df_calc[prediction_col] == 'A'
    ]
    choices = [df_calc['HomeOdds'], df_calc['DrawOdds'], df_calc['AwayOdds']]
    df_calc['TakenOdds'] = np.select(conditions, choices, default=1.0)
    
    df_calc['Won'] = df_calc[prediction_col] == df_calc[target_col]
    taken_odds = pd.to_numeric(df_calc['TakenOdds'], errors='coerce').fillna(1.0)
    df_calc['PnL'] = np.where(df_calc['Won'], STAKE * (taken_odds - 1), -STAKE)
    
    df_calc['Date'] = pd.to_datetime(df_calc['Date'], errors='coerce')
    df_calc = df_calc.dropna(subset=['Date'])
    df_calc['YearWeek'] = df_calc['Date'].dt.to_period('W')
    
    weekly = df_calc.groupby('YearWeek').agg({'PnL': 'sum', 'Date': 'count'}).rename(columns={'Date': 'Bets'})
    weekly = weekly[weekly['Bets'] >= MIN_BETS_PER_WEEK]
    
    if len(weekly) == 0: return pd.DataFrame()
    weekly[strategy_name] = (weekly['PnL'] / (weekly['Bets'] * STAKE)) * 100
    return weekly[[strategy_name]].astype(float)

def reconstruct_benchmarks(master_df):
    df = master_df.copy()
    df['Pred_Home'] = 'H'
    odds_cols = ['HomeOdds', 'DrawOdds', 'AwayOdds']
    df['Pred_Favorite'] = df[odds_cols].idxmin(axis=1).map({'HomeOdds': 'H', 'DrawOdds': 'D', 'AwayOdds': 'A'}, na_action='ignore')
    df['Pred_Underdog'] = df[odds_cols].idxmax(axis=1).map({'HomeOdds': 'H', 'DrawOdds': 'D', 'AwayOdds': 'A'}, na_action='ignore')
    
    def big_six_logic(row):
        h = row.get('HomeTeam', '') in BIG_SIX
        a = row.get('AwayTeam', '') in BIG_SIX
        if h and not a: return 'H'
        if a and not h: return 'A'
        return None 
    df['Pred_BigSix'] = df.apply(big_six_logic, axis=1)
    return df

def run_comprehensive_test():
    print("----------------------------------------------------------------")
    print("Script: statistical_significance_test.py")
    print("Task:   ROI T-Tests & Diebold-Mariano Forecast Analysis")
    print("----------------------------------------------------------------")

    if not os.path.exists(MASTER_FILE):
        print(f"Error: {MASTER_FILE} not found.")
        return

    print("Loading Master Data...")
    df_master = pd.read_csv(MASTER_FILE)
    df_master['Date'] = pd.to_datetime(df_master['Date'], errors='coerce')
    for c in ['HomeOdds', 'DrawOdds', 'AwayOdds']:
        df_master[c] = pd.to_numeric(df_master[c], errors='coerce')
    df_master = df_master.dropna(subset=['Date', 'HomeOdds', 'AwayOdds'])
    
    print("Reconstructing Benchmark Strategies...")
    df_master = reconstruct_benchmarks(df_master)
    
    bench_series = {}
    bench_names = ['Pred_Favorite', 'Pred_Home', 'Pred_Underdog', 'Pred_BigSix']
    for b_col in bench_names:
        clean_name = b_col.replace('Pred_', 'Bench_')
        series = get_weekly_roi_series(df_master, b_col, clean_name)
        if not series.empty: bench_series[clean_name] = series

    model_files = glob.glob("results_*.csv")
    # Filter out heuristic results, keep only ML results
    model_files = [f for f in model_files if "heuristic" not in f and "results_" in f]
    
    if not model_files:
        print("No ML result files found.")
        return

    report_lines = []
    report_lines.append("COMPREHENSIVE STATISTICAL REPORT")
    report_lines.append("===================================================\n")
    
    all_plot_data = pd.DataFrame()

    for m_file in model_files:
        model_name = os.path.basename(m_file).replace("results_", "").replace(".csv", "")
        print(f"\nProcessing Model: {model_name}")
        
        df_model_raw = pd.read_csv(m_file)
        # Ensure numerics
        for c in ['HomeOdds', 'HomeProb', 'DrawProb', 'AwayProb']:
            if c in df_model_raw.columns:
                df_model_raw[c] = pd.to_numeric(df_model_raw[c], errors='coerce')

        # --- 1. ROI T-TESTS ---
        ml_series = get_weekly_roi_series(df_model_raw, 'Predicted', model_name)
        
        if len(ml_series) < 10:
            print(f"  -> Skipping {model_name} (Not enough data)")
            continue

        report_lines.append(f"MODEL: {model_name}")
        report_lines.append("-" * 60)
        
        # --- 2. DIEBOLD-MARIANO TEST (Forecast Accuracy) ---
        # Calculate Benchmark (Market) Probabilities
        df_model_raw['Imp_H'] = 1 / df_model_raw['HomeOdds']
        df_model_raw['Imp_D'] = 1 / df_model_raw['DrawOdds']
        df_model_raw['Imp_A'] = 1 / df_model_raw['AwayOdds']
        total_imp = df_model_raw['Imp_H'] + df_model_raw['Imp_D'] + df_model_raw['Imp_A']
        df_model_raw['Bench_Prob_H'] = df_model_raw['Imp_H'] / total_imp

        # Compare Model Home Prob vs Market Home Prob
        # Target: 1 if Home Win, 0 if not
        if 'Actual' in df_model_raw.columns and 'HomeProb' in df_model_raw.columns:
            y_true_dm = (df_model_raw['Actual'] == 'H').astype(int)
            dm_stat, dm_p = diebold_mariano_test(y_true_dm, df_model_raw['HomeProb'], df_model_raw['Bench_Prob_H'])
            
            report_lines.append("FORECAST ACCURACY (Diebold-Mariano Test)")
            report_lines.append(f"   vs Efficient Market (Odds): DM Stat={dm_stat:.4f}, P-Value={dm_p:.4e}")
            if dm_p > 0.05:
                report_lines.append("   CONCLUSION: Model is STATISTICALLY INDISTINGUISHABLE from Market Odds.")
            else:
                report_lines.append("   CONCLUSION: Model is distinct (statistically significant difference).")
            report_lines.append("-" * 60)

        # ROI Benchmark Table
        report_lines.append(f"{'BENCHMARK':<20} | {'N (Wks)':<8} | {'DIFF':<8} | {'P-VAL (2-Sided)':<15} | {'P-VAL (Superior)':<15} | {'CONCLUSION'}")
        
        for b_name, b_series in bench_series.items():
            merged = pd.merge(ml_series, b_series, left_index=True, right_index=True)
            merged = merged.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(merged) < 30: continue
            
            vec_ml = merged[model_name].values
            vec_bench = merged[b_name].values
            
            t_stat, p_two = stats.ttest_rel(vec_ml, vec_bench)
            p_one = p_two / 2 if t_stat > 0 else 1.0 - (p_two / 2)
            mean_diff = np.mean(vec_ml) - np.mean(vec_bench)
            
            if p_one < 0.05: concl = "AI WINS (Sig)"
            elif p_one < 0.10: concl = "AI WINS (Marginal)"
            else: concl = "No Edge"
            
            line = f"{b_name:<20} | {len(merged):<8} | {mean_diff:>+6.2f}% | {p_two:.5f}         | {p_one:.5f}         | {concl}"
            report_lines.append(line)

            if b_name == 'Bench_Favorite':
                merged['Model'] = model_name
                merged['Benchmark_ROI'] = merged[b_name]
                merged['Model_ROI'] = merged[model_name]
                all_plot_data = pd.concat([all_plot_data, merged])

        report_lines.append("\n")

    with open("final_statistical_report.txt", "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nSaved detailed analysis to 'final_statistical_report.txt'")
    
    # Plotting Logic (Unchanged)
    if not all_plot_data.empty:
        plt.figure(figsize=(16, 9))
        sns.set_style("whitegrid")
        models = all_plot_data['Model'].unique()
        for m in models:
            subset = all_plot_data[all_plot_data['Model'] == m].sort_index()
            cum_model = subset['Model_ROI'].cumsum()
            dates = subset.index.to_timestamp()
            plt.plot(dates, cum_model.values, label=f"{m} Model", linewidth=2.5)
            
        first_m = models[0]
        subset = all_plot_data[all_plot_data['Model'] == first_m].sort_index()
        cum_bench = subset['Benchmark_ROI'].cumsum()
        dates_bench = subset.index.to_timestamp()
        
        plt.plot(dates_bench, cum_bench.values, label="Market Benchmark (Bet on Favorite)", 
                 color='black', linestyle='--', linewidth=2, alpha=0.7)
        
        plt.title("Cumulative Wealth Accumulation: Machine Learning vs Efficient Market Baseline", fontsize=18, pad=20)
        plt.xlabel("Timeline (Years)", fontsize=14, labelpad=15)
        plt.ylabel("Cumulative ROI Points", fontsize=14, labelpad=15)
        plt.legend(fontsize=12, loc='upper left', frameon=True, framealpha=0.9)
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(fontsize=12, rotation=45)
        plt.tight_layout()
        plt.savefig("fig_cumulative_wealth.png", dpi=300)
        print("Saved 'fig_cumulative_wealth.png'")

if __name__ == "__main__":
    run_comprehensive_test()
