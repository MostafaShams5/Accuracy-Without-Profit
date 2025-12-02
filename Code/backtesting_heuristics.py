"""
Script: backtesting_heuristics.py
Author: Mostafa Shams
Description:
    This script establishes the "Human Baseline" for the study "Testing the Semi-Strong
    Efficient Market Hypothesis in Sports Betting."

    It simulates five distinct betting strategies that represent common market behaviors,
    running them on the exact same dataset as the machine learning models to ensure a
    fair, apples-to-apples comparison.

    Strategies Evaluated:
    1.  The "Form" Bettor (Momentum Strategy):
        - Logic: Bets on the team with the superior recent record (Last 5 Games).
        - Rule: If Home_Avg_Points_L5 > Away_Avg_Points_L5, Bet Home.
                If Away_Avg_Points_L5 > Home_Avg_Points_L5, Bet Away.
                If equal, no bet is placed (simulating a 'wait-and-see' approach).

    2.  The "Big Six" Bias (Prestige Strategy):
        - Logic: Simulates the casual bettor's bias toward the historically dominant clubs
          (Arsenal, Chelsea, Liverpool, Man Utd, Man City, Tottenham).
        - Rule: If a "Big Six" team plays a non-"Big Six" team at Home, Bet Home.
                If a "Big Six" team plays a non-"Big Six" team Away, Bet Away.
                Matches between two "Big Six" teams are skipped (too risky/efficient).

    3.  The "Market Consensus" (Efficient Market Baseline):
        - Logic: Always trusts the bookmaker's pricing.
        - Rule: Bet on the outcome with the lowest numerical odds (The Favorite).

    4.  The "Contrarian" (Underdog Strategy):
        - Logic: Attempts to exploit the "Favorite-Longshot Bias" by betting against the public.
        - Rule: Bet on the outcome with the highest numerical odds.

    5.  The "Home Field" Bias (Naive Baseline):
        - Logic: Simply bets on the Home Team every time, exploiting the known home advantage.

    Output:
        - 'report_heuristics_summary.txt': Detailed financial metrics (ROI, Win Rate, PnL).
        - 'fig_heuristics_roi.png': A comparative line chart of ROI evolution over time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as ticker

INPUT_FILE = "master_feature_table.csv"
OUTPUT_REPORT = "report_heuristics_summary.txt"
INITIAL_BANKROLL = 10000
STAKE = 100

BIG_SIX = ['Arsenal', 'Chelsea', 'Liverpool', 'Man United', 'Man City', 'Tottenham']

def load_data():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"CRITICAL: {INPUT_FILE} not found. Run process_football_data.py first.")
    
    df = pd.read_csv(INPUT_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    df['Season'] = np.where(df['Date'].dt.month >= 8, df['Date'].dt.year, df['Date'].dt.year - 1)
    
    cols = ['HomeOdds', 'DrawOdds', 'AwayOdds']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = df.dropna(subset=cols)
    return df

def strategy_form_bettor(row):
    hp = row.get('Home_Avg_Points_L5', 0)
    ap = row.get('Away_Avg_Points_L5', 0)
    
    if hp > ap:
        return 'H'
    elif ap > hp:
        return 'A'
    return None

def strategy_big_six(row):
    home_is_big = row['HomeTeam'] in BIG_SIX
    away_is_big = row['AwayTeam'] in BIG_SIX
    
    if home_is_big and not away_is_big:
        return 'H'
    elif away_is_big and not home_is_big:
        return 'A'
    return None

def strategy_favorite(row):
    odds = {'H': row['HomeOdds'], 'D': row['DrawOdds'], 'A': row['AwayOdds']}
    return min(odds, key=odds.get)

def strategy_underdog(row):
    odds = {'H': row['HomeOdds'], 'D': row['DrawOdds'], 'A': row['AwayOdds']}
    return max(odds, key=odds.get)

def strategy_home_team(row):
    return 'H'

def run_backtest(df):
    strategies = {
        "Form Bettor": strategy_form_bettor,
        "Big Six Bias": strategy_big_six,
        "Bet on Favorite": strategy_favorite,
        "Bet on Underdog": strategy_underdog,
        "Bet Home Team": strategy_home_team
    }
    
    results = []
    
    print(f"Backtesting {len(strategies)} heuristic strategies on {len(df)} matches...")
    
    for name, func in strategies.items():
        print(f"  -> Simulating: {name}")
        
        bets = df.apply(func, axis=1)
        
        df_res = df.copy()
        df_res['Prediction'] = bets
        
        df_active = df_res.dropna(subset=['Prediction']).copy()
        
        df_active['Won'] = df_active['Prediction'] == df_active['FTR']
        
        conditions = [
            df_active['Prediction'] == 'H',
            df_active['Prediction'] == 'D',
            df_active['Prediction'] == 'A'
        ]
        choices = [df_active['HomeOdds'], df_active['DrawOdds'], df_active['AwayOdds']]
        df_active['BetOdds'] = np.select(conditions, choices, default=1)
        
        df_active['PnL'] = np.where(df_active['Won'], STAKE * (df_active['BetOdds'] - 1), -STAKE)
        
        total_bets = len(df_active)
        if total_bets > 0:
            accuracy = df_active['Won'].mean()
            total_pnl = df_active['PnL'].sum()
            roi = (total_pnl / (total_bets * STAKE)) * 100
            
            seasonal = df_active.groupby('Season')['PnL'].sum()
            bet_counts = df_active.groupby('Season')['Date'].count()
            seasonal_roi = (seasonal / (bet_counts * STAKE)) * 100
            
            safe_name = name.replace(" ", "_")
            seasonal_roi_df = seasonal_roi.reset_index(name='ROI')
            seasonal_roi_df.to_csv(f"results_heuristic_{safe_name}.csv", index=False)

            results.append({
                'Strategy': name,
                'Accuracy': accuracy,
                'ROI': roi,
                'Total_PnL': total_pnl,
                'Bets_Placed': total_bets,
                'Seasonal_ROI': seasonal_roi
            })
        else:
            print(f"     [WARNING] Strategy {name} placed 0 bets.")

    return results

def plot_results(results):
    plt.figure(figsize=(16, 9))
    
    for res in results:
        series = res['Seasonal_ROI']
        plt.plot(series.index, series.values, marker='o', label=res['Strategy'])
        
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
    plt.title("Heuristic Strategies: Year-Over-Year ROI Analysis")
    plt.xlabel("Season")
    plt.ylabel("ROI (%)")
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("fig_heuristics_roi.png")
    plt.close()

def main():
    print("----------------------------------------------------------------")
    print("Script: backtesting_heuristics.py")
    print("Author: Mostafa Shams")
    print("Task:   Establishing Human/Market Benchmarks")
    print("----------------------------------------------------------------")
    
    df = load_data()
    results = run_backtest(df)
    
    plot_results(results)
    
    print("\n--- HEURISTIC PERFORMANCE SUMMARY ---")
    summary_data = [{k: v for k, v in r.items() if k != 'Seasonal_ROI'} for r in results]
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    with open(OUTPUT_REPORT, "w") as f:
        f.write("HEURISTIC STRATEGIES REPORT\n")
        f.write("===========================\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n ANALYSIS:\n")
        f.write(" - 'Bet on Favorite' typically has high Accuracy but negative ROI (The Favorite-Longshot Bias).\n")
        f.write(" - 'Bet on Underdog' typically has high variance and highly negative ROI.\n")
        f.write(" - 'Big Six Bias' tests if blindly following prestige pays off.\n")
        
    print(f"\nReport saved to {OUTPUT_REPORT}")
    print("Figures saved to fig_heuristics_roi.png")

if __name__ == "__main__":
    main()
