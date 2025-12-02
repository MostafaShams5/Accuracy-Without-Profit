"""
Script: simulation_parametric_monte_carlo.py
Author: Mostafa Shams
Description:
    This script performs the final Parametric Monte Carlo Simulation for the study
    "Testing the Semi-Strong Efficient Market Hypothesis in Sports Betting."

    It combines the rigorous empirical data derived from Walk-Forward Validation
    with high-fidelity stochastic modeling.

    Methodology:
    1.  Empirical Inputs: Seeding the simulation with the exact Win Rates and ROIs
        found in the backtesting phase (e.g., XGBoost +0.29%).
    2.  Vectorized Simulation: Simulating 20,000 lifetimes (5,000 bets each) using
        Binomial distributions for computational efficiency.
    3.  Advanced Risk Metrics: Calculating Value at Risk (VaR 95%), Sortino Ratio,
        and Ruin Probability to assess tradeability.

    Output:
        - 'report_final_monte_carlo.txt': A detailed financial report for every strategy.
        - 'plot_hist_[strategy_name].png': High-resolution distribution plots matching
          the original project's aesthetic style.
        - 'plot_comparative_kde.png': An overlay comparing AI vs. Human strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import time
from scipy.stats import norm

# --- Configuration ---
OUTPUT_REPORT_FILE = 'report_final_monte_carlo.txt'
NUM_SIMULATIONS = 20000
INITIAL_BANKROLL = 10000
BETS_PER_SIMULATION = 5000
STAKE_PER_BET = 100

# --- EMPIRICAL INPUTS (From Walk-Forward Validation) ---
STRATEGIES_DATA = {
    "XGBoost (AI)":       {"win_prob": 0.5107, "roi": 0.0029},
    "LightGBM (AI)":      {"win_prob": 0.5179, "roi": -0.0111},
    "Random Forest (AI)": {"win_prob": 0.5283, "roi": -0.0201},
    "Big Six Bias":       {"win_prob": 0.6045, "roi": -0.0251},
    "Bet on Favorite":    {"win_prob": 0.5357, "roi": -0.0414},
    "Form Bettor":        {"win_prob": 0.4795, "roi": -0.0265},
    "Bet Home Team":      {"win_prob": 0.4688, "roi": -0.0242},
    "Bet Underdog":       {"win_prob": 0.2133, "roi": -0.0819}
}

def calculate_implied_odds(win_prob, roi):
    if win_prob <= 0: return 0
    return (roi + 1) / win_prob

def run_simulation(strategy_name, params):
    win_prob = params['win_prob']
    roi = params['roi']
    odds = calculate_implied_odds(win_prob, roi)
    
    print(f"  -> Simulating: {strategy_name:<20} | Implied Odds: {odds:.2f} | Win Prob: {win_prob:.1%}")
    
    profit_per_win = STAKE_PER_BET * (odds - 1)
    loss_per_loss = STAKE_PER_BET
    
    # Vectorized Simulation
    bankrolls = np.full(NUM_SIMULATIONS, float(INITIAL_BANKROLL))
    wins = np.random.binomial(BETS_PER_SIMULATION, win_prob, NUM_SIMULATIONS)
    losses = BETS_PER_SIMULATION - wins
    net_pnl = (wins * profit_per_win) - (losses * loss_per_loss)
    
    final_bankrolls = bankrolls + net_pnl
    return final_bankrolls, odds

def generate_detailed_plot(bankrolls, strategy_name):
    """
    Generates the high-fidelity histogram matching the original project style.
    """
    filename = f"plot_hist_{strategy_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    print(f"     [Plotting] Generating detailed histogram -> {filename}")
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(20, 11), facecolor='#f8f9fa')
    ax.set_facecolor('#f8f9fa')
    
    # The specific dark blue color from your original code
    sns.histplot(bankrolls, bins=200, color='#001019', alpha=0.75, kde=False, element="step", fill=True, ax=ax)

    mean_val = bankrolls.mean()
    percent_won = np.sum(bankrolls > INITIAL_BANKROLL) / NUM_SIMULATIONS * 100
    percent_bankrupt = np.sum(bankrolls <= 0) / NUM_SIMULATIONS * 100

    # Lines
    ax.axvline(INITIAL_BANKROLL, color='#d62828', linestyle='--', linewidth=2.5, label='Starting Bankroll')
    ax.axvline(mean_val, color='#f77f00', linestyle='-', linewidth=2.5, label='Average Final Bankroll')
    
    # Annotation 1: Parameters
    params_text = (f"$\\bf{{Simulation\ Parameters}}$\n"
                   f"Initial Bankroll: ${INITIAL_BANKROLL:,}\n"
                   f"Bet Stake: ${STAKE_PER_BET}\n"
                   f"bets/Sim: {BETS_PER_SIMULATION:,}\n"
                   f"───────────────\n"
                   f"$\\bf{{Key\ Values}}$\n"
                   f"Avg Final: ${mean_val:,.0f}\n")
    
    ax.text(0.99, 0.99, params_text, transform=ax.transAxes, fontsize=12, va='top', ha='right', 
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.9, ec='#cccccc'))

    # Annotation 2: Segmentation
    segmentation_text = (f"$\\bf{{Population\ Segmentation}}$\n"
                         f"• Profitable: {percent_won:.1f}%\n"
                         f"• Unprofitable: {(100 - percent_won - percent_bankrupt):.1f}%\n"
                         f"• Ruin (<0): {percent_bankrupt:.1f}%")
    
    ax.text(0.02, 0.99, segmentation_text, transform=ax.transAxes, fontsize=12, va='top', 
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.9, ec='#cccccc'))

    fig.suptitle(f'Monte Carlo Simulation: "{strategy_name}" Strategy', fontsize=22, fontweight='bold', color='#333333')
    ax.set_title(f'Final bankroll distribution for {NUM_SIMULATIONS:,} entities after {BETS_PER_SIMULATION} bets', fontsize=14, pad=10)
    
    ax.set_xlabel('Final Bankroll ($)', fontsize=14, labelpad=10)
    ax.set_ylabel('Frequency', fontsize=14, labelpad=10)
    
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${int(x):,}'))
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    sns.despine(left=True, bottom=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def generate_comparative_plot(all_results):
    """
    Generates an overlay KDE plot to compare the AI vs the House Edge.
    """
    print("     [Plotting] Generating comparative KDE plot -> plot_comparative_kde.png")
    plt.figure(figsize=(16, 9), facecolor='white')
    sns.set_style("whitegrid")
    
    # Key strategies to compare
    targets = ["XGBoost (AI)", "Bet Home Team", "Bet on Favorite"]
    colors = {"XGBoost (AI)": "#2ca02c", "Bet Home Team": "#d62828", "Bet on Favorite": "#1f77b4"}
    
    for name in targets:
        if name in all_results:
            sns.kdeplot(all_results[name], label=name, fill=True, alpha=0.1, linewidth=2.5, color=colors.get(name, 'gray'))
            
    plt.axvline(INITIAL_BANKROLL, color='black', linestyle='--', label="Break-Even", linewidth=1.5)
    
    plt.title(f"Comparative Performance Density ({BETS_PER_SIMULATION} Bets)", fontsize=18, fontweight='bold')
    plt.xlabel("Final Bankroll ($)", fontsize=14)
    plt.ylabel("Probability Density", fontsize=14)
    plt.legend(fontsize=12, title="Strategy")
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig("plot_comparative_kde.png", dpi=300)
    plt.close()

def generate_report_section(bankrolls, strategy_name, win_prob, odds):
    """
    Calculates detailed financial metrics (VaR, Sortino, etc.) for the text report.
    """
    ev_per_bet = (win_prob * (STAKE_PER_BET * (odds - 1))) - ((1 - win_prob) * STAKE_PER_BET)
    theoretical_mean = INITIAL_BANKROLL + (ev_per_bet * BETS_PER_SIMULATION)
    
    mean_val = bankrolls.mean()
    median_val = np.median(bankrolls)
    std_dev = bankrolls.std()
    
    # ROI calculation based on total turnover
    total_turnover = BETS_PER_SIMULATION * STAKE_PER_BET
    avg_profit = mean_val - INITIAL_BANKROLL
    roi_percent = (avg_profit / total_turnover) * 100
    
    # Risk Metrics
    # Sortino Ratio (approximate for final distribution)
    returns = (bankrolls - INITIAL_BANKROLL) / INITIAL_BANKROLL
    downside = returns[returns < 0]
    downside_dev = downside.std() if len(downside) > 0 else 1e-9
    sortino = returns.mean() / downside_dev if downside_dev > 0 else 0
    
    # Value at Risk (95%)
    var_95_absolute = np.percentile(bankrolls, 5)
    var_95_loss = INITIAL_BANKROLL - var_95_absolute

    profitable_count = np.sum(bankrolls > INITIAL_BANKROLL)
    bankrupt_count = np.sum(bankrolls <= 0)
    
    report = [
        f"\n{'='*85}",
        f"STRATEGY ANALYSIS: {strategy_name.upper()}",
        f"{'='*85}",
        "## Mathematical Expectation (Inputs)",
        f"  - True Win Probability: {win_prob:.2%}",
        f"  - Implied Payout Odds:  {odds:.4f}",
        f"  - Expected Value (EV):  ${ev_per_bet:.4f} per bet",
        f"  - Expected ROI:         {roi_percent:.2f}%",
        "\n## Simulation Results (20,000 Lifetimes)",
        f"  - Average Final Bankroll: ${mean_val:,.2f}",
        f"  - Median Final Bankroll:  ${median_val:,.2f}",
        f"  - Standard Deviation:     ${std_dev:,.2f}",
        "\n## Risk Profile",
        f"  - Risk of Ruin (Bankrupt): {bankrupt_count:,} players ({bankrupt_count/NUM_SIMULATIONS:.1%})",
        f"  - Probability of Profit:   {profitable_count/NUM_SIMULATIONS:.1%}",
        f"  - 95% Value at Risk (VaR): In the worst 5% of cases, you end with <= ${var_95_absolute:,.2f}",
        f"  - Sortino Ratio:           {sortino:.4f}"
    ]
    return "\n".join(report)

def main():
    print("----------------------------------------------------------------")
    print("Script: simulation_parametric_monte_carlo.py")
    print("----------------------------------------------------------------")
    
    full_report = [
        f"{'='*85}",
        "        PARAMETRIC MONTE CARLO: COMPREHENSIVE STATISTICAL REPORT",
        f"{'='*85}\n",
        "## Global Simulation Parameters",
        f"  - Population: {NUM_SIMULATIONS:,} independent simulations",
        f"  - Timeline:   {BETS_PER_SIMULATION:,} bets per simulation",
        f"  - Bankroll:   ${INITIAL_BANKROLL:,.0f} start / ${STAKE_PER_BET} fixed stake"
    ]
    
    all_results = {}
    
    for name, params in STRATEGIES_DATA.items():
        bankrolls, odds = run_simulation(name, params)
        all_results[name] = bankrolls
        
        # Add section to text report
        full_report.append(generate_report_section(bankrolls, name, params['win_prob'], odds))
        
        # Generate individual detailed plot for key strategies
        # We plot XGBoost (The Winner) and Bet Home Team (The Benchmark)
        if name in ["XGBoost (AI)", "Bet Home Team"]:
            generate_detailed_plot(bankrolls, name)
            
    # Generate Comparison Plot
    generate_comparative_plot(all_results)
    
    # Save Report
    print(f"\n[Report] Saving detailed statistics to {OUTPUT_REPORT_FILE}...")
    with open(OUTPUT_REPORT_FILE, "w") as f:
        f.write("\n".join(full_report))
        
    print("Done. All outputs generated successfully.")

if __name__ == "__main__":
    main()
