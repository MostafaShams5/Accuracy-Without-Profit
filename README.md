
# Accuracy Without Profit: A Statistical Evaluation of Machine Learning in the EPL

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17794277.svg)](https://doi.org/10.5281/zenodo.17794277)

This repository contains the complete analysis pipeline, datasets, and simulation code for the research paper: **"Accuracy Without Profit: A Statistical Evaluation of Machine Learning in the English Premier League Betting Market."**

This is the official repository for the paper, as cited in the publication.

---

## The Core Thesis

In data science, we optimize for **Accuracy** or **AUC**. In financial markets, those metrics are often meaningless.

This study explores the **"Accuracy Paradox"**: the phenomenon where the model with the highest predictive accuracy generates the largest financial loss. By training models on 21 years of English Premier League data (2000–2021) and benchmarking them against a control group of pure noise, we isolate the exact moment modern markets became "efficient" and quantify how AI overestimates its own intelligence.

---

## Part I: The Control Group (Live Web Scraping & Chaos)

Before modeling football, we needed a baseline of "unpredictability" to ensure our deep learning models weren't hallucinating patterns.

### 1. Data Acquisition
We built a custom **Playwright-based web scraper** to target the **1xBet** "Crash" game. Unlike static datasets, this allowed us to harvest 2,596 rounds of live, real-time gambling data.

### 2. The "Sanity Check"
We applied the **Augmented Dickey-Fuller (ADF)** test to this time-series data. The result ($p \approx 1.06 \times 10^{-29}$) proved the data was stationary and purely random.

### 3. Neural Network Validation
We trained **LSTM Regressors** and **LSTM Autoencoders** on sliding windows ($T=15$) of this data.
*   **The Result:** The models failed to converge on a loss lower than the baseline.
*   **The Takeaway:** This was a critical success. It proved our architecture was robust enough to ignore pure noise, validating that any signal found later in the football data was genuine.

---

## Part II: The Feature Engineering & Models

We treated the Premier League as a multi-class classification problem ($Home, Draw, Away$), but with a strict temporal constraint to mimic real-world betting.

### The "Rolling Window" Technique
Raw match stats (e.g., "Arsenal had 5 shots") are noisy. To capture **"Team Form,"** we engineered rolling averages over the **Last 5 Games (L5)**:
*   **Momentum:** Rolling Points Average.
*   **Defensive Form:** Rolling Goals Conceded.
*   **Offensive Threat:** Rolling Shots on Target & Corners.

### The "Walk-Forward" Protocol
Most studies "cheat" by shuffling data. We used **Chronological Walk-Forward Validation**.
*   *Train:* 2000–2005 $\rightarrow$ *Predict:* 2006.
*   *Train:* 2000–2006 $\rightarrow$ *Predict:* 2007.
The model was never allowed to see a single data point from the future, preserving the integrity of the backtest.

### The Algorithms
We optimized three models using specific objective functions for probability estimation:
1.  **XGBoost:** `multi:softprob` (for precise probability calibration).
2.  **LightGBM:** Gradient Boosting Decision Trees.
3.  **Random Forest:** Bagging ensemble (100 trees).

---

## Part III: The Findings

### 1. The Accuracy Paradox
The **Random Forest** model achieved the highest raw accuracy (**52.83%**), yet it generated the largest financial loss (**-$11,049**).
Conversely, **XGBoost** had the lowest accuracy (**51.06%**) but was the only model to generate a profit (**+$1,611**).
*   **Insight:** Optimizing for frequency of wins (Accuracy) leads to betting on "safe" favorites with poor value. Optimizing for probability (LogLoss) is the only path to profit.

### 2. The "Draw" Blind Spot
Analysis of the Confusion Matrices revealed a systemic flaw: the models were **Risk Averse**. They effectively refused to predict "Draws," chasing the Home Team advantage instead. By ignoring the high-variance/high-reward "Draw" outcome, the models artificially inflated their accuracy while destroying their profitability.

### 3. The Alpha Decay (2015 was the Death of the Edge)
We mapped the ROI over time, revealing two distinct eras:
*   **The Golden Age (2006–2014):** The model consistently beat the bookmaker.
*   **The Efficient Era (2015–2021):** The edge evaporated.
A linear regression of the returns shows a negative slope of **-0.51**, proving the **Adaptive Market Hypothesis**: as bookmakers integrated better algorithms around 2015, the "simple" inefficiencies our models exploited were priced out of the market.

### 4. SHAP Analysis: Playing the Bookie, Not the Game
Using **SHAP (SHapley Additive exPlanations)**, we opened the "Black Box."
*   **Top Feature:** `HomeOdds` (0.33 importance).
*   **Second Feature:** `AwayOdds` (0.24 importance).
*   **Team Stats:** `Avg_Shots` (0.13 importance).
The AI learned that the Bookmaker's Price was a better predictor of the outcome than the team's actual performance statistics.

### 5. The Calibration Trap (Why Kelly Failed)
We tested two money-management strategies: **Flat Betting** (betting \$100 every time) vs. the **Kelly Criterion** (betting based on confidence).
*   **Flat Result:** \$11,611.
*   **Kelly Result:** \$9,206.
*   **The Cause:** We measured an **Expected Calibration Error (ECE)** of **11%**. The models were chronically overconfident. When the AI said, "I am 60% sure," it was only right 49% of the time. This "arrogance" caused the Kelly strategy to bet too big and lose capital.

---

## Part IV: Risk & Ruin

We ran **20,000 Monte Carlo simulations** to model the life of a bettor using these AI strategies.
*   **Risk of Ruin:** 5.1% of all simulated players went completely bankrupt.
*   **Unprofitable Lifetimes:** 36.6% of players ended the 15-year period with less money than they started.
*   **Sortino Ratio:** A modest 0.38, indicating that the returns did not justify the downside volatility.

---

## How to Cite

This repository supports the findings in the following paper. If you use this code, the scraped datasets, or the feature engineering pipelines, please cite:

**APA Style**
> Shams, M. (2025). Accuracy Without Profit: A Statistical Evaluation of Machine Learning in the English Premier League Betting Market. Zenodo. https://doi.org/10.5281/zenodo.17794277

**BibTeX**
```bibtex
@article{shams_2025_accuracy,
  author       = {Shams, Mostafa},
  title        = {Accuracy Without Profit: A Statistical Evaluation of Machine Learning in the English Premier League Betting Market},
  publisher    = {Zenodo},
  year         = {2025},
  doi          = {10.5281/zenodo.17794277},
  url          = {https://doi.org/10.5281/zenodo.17794277}
}
```

---

*License: Creative Commons Attribution 4.0 International (CC BY 4.0)*
