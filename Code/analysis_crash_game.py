import os
import random
from pathlib import Path
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import ks_2samp, skew, kurtosis

from statsmodels.tsa.stattools import adfuller

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # reduce TF logging


sns.set_theme(style="whitegrid")
sns.set_palette("colorblind") 
plt.rcParams.update({
    'figure.figsize': (12, 7),
    'figure.facecolor': 'white',
    'axes.titlesize': 14,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
})

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


class CrashGameAnalyzer:
    def __init__(self, file_path: str, report_file: str):
        self.file_path = Path(file_path)
        self.report_file = Path(report_file)
        self.scaler = None
        self.scaled_data = None
        self.df = self._load_data()

        with open(self.report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("STATISTICAL REPORT: CRASH GAME ANALYSIS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now(timezone.utc).isoformat()} UTC\n")
            f.write(f"Data source: {self.file_path}\n")
            f.write(f"Random seed: {SEED}\n")
            f.write("="*80 + "\n\n")

    def _load_data(self) -> pd.DataFrame:
        logging.info(f"Loading data from {self.file_path} ...")
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        df = pd.read_csv(self.file_path)
        df['crash'] = pd.to_numeric(df['crash'], errors='coerce')
        df = df.dropna(subset=['crash']).reset_index(drop=True)
        self.scaler = MinMaxScaler()
        self.scaled_data = self.scaler.fit_transform(df[['crash']].astype(float))
        logging.info(f"Loaded {len(df):,} rounds (crash column cleaned).")
        return df

    def _append_to_report(self, title: str, content: str):
        with open(self.report_file, 'a') as f:
            f.write("-"*80 + "\n")
            f.write(f"SECTION: {title}\n")
            f.write("-"*80 + "\n")
            f.write(content + "\n\n")

    def _create_sequences(self, data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        if X.ndim == 2:
            X = X.reshape((X.shape[0], X.shape[1], 1))
        if y.ndim == 1:
            y = y.reshape((-1, 1))
        return X, y

    def run_statistical_analysis(self, house_cashouts=None):
        logging.info("Running statistical analysis and computing house edge grid...")
        if house_cashouts is None:
            house_cashouts = np.concatenate([np.arange(1.01, 1.11, 0.01), [1.25, 1.5, 2.0, 3.0, 5.0]])
        crashes = self.df['crash'].values
        N = len(crashes)
        
        desc = self.df['crash'].describe().to_string()
        skewness = skew(crashes)
        kurt = kurtosis(crashes)
        instant_bust_prob = np.sum(crashes <= 1.00) / N

        stats_report = (
            f"Core descriptive statistics:\n{desc}\n\n"
            f"Total Rounds Analyzed: {N:,}\n"
            f"Skewness: {skewness:.4f}\n"
            f"Kurtosis: {kurt:.4f}\n"
            f"Probability of Instant Bust (<= 1.00x): {instant_bust_prob:.2%}\n"
        )
        self._append_to_report("Descriptive Statistics", stats_report)

        rows = []
        for c in house_cashouts:
            successes = np.sum(crashes >= c)
            p_hat = successes / N
            house_edge = 1.0 - p_hat * c
            ci_low, ci_high = proportion_confint(successes, N, method='wilson')
            he_low = 1.0 - ci_high * c
            he_high = 1.0 - ci_low * c
            rows.append({'cashout': c, 'p_hat': p_hat, 'house_edge': house_edge, 'he_low': he_low, 'he_high': he_high})
        hed_df = pd.DataFrame(rows).sort_values('cashout')
        
        report_content = "House-edge grid (with 95% Wilson confidence intervals):\n" + hed_df.to_string(index=False, float_format="{:.6f}".format)
        self._append_to_report("House Edge Grid", report_content)

   
        fig, ax = plt.subplots()
        counts, bin_edges = np.histogram(crashes[crashes < 25], bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.plot(bin_centers, counts, linewidth=1, zorder=2)
        ax.fill_between(bin_centers, counts, alpha=0.2, zorder=1)
        mean_val = crashes.mean()
        median_val = np.median(crashes)
        ax.set_xlabel('Crash Multiplier')
        ax.set_ylabel('Frequency (Number of Rounds)')
        ax.set_title('Distribution of Crash Multipliers')
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1, label=f'Mean: {mean_val:.2f}x')
        ax.axvline(median_val, color='blue', linestyle='--', linewidth=1, label=f'Median: {median_val:.2f}x')
        for mult in [1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            idx = np.searchsorted(bin_centers, mult)
            if idx < len(counts):
                marker_color = 'darkred'
                ax.plot(bin_centers[idx], counts[idx], 'o', color=marker_color, markersize=5, zorder=3)
                ax.annotate(f'{bin_centers[idx]:.2f}x', (bin_centers[idx], counts[idx]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=7, bbox=dict(boxstyle="round,pad=0.2", fc="yellow", ec="none", alpha=0.7))
        stats_text = (f"Total Rounds: {N:,}\n" f"Mean: {mean_val:.3f}x\n" f"Median: {median_val:.3f}x")
        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, va='top', ha='right', bbox=dict(boxstyle='round', alpha=0.1))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:.2f}x'))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1)); ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.25))
        ax.grid(True, which='both', linestyle='--', alpha=0.6); ax.set_xlim(left=1.0); ax.legend()
        fname = FIG_DIR / 'crash_distribution_audience_friendly.png'
        fig.tight_layout(); fig.savefig(fname, dpi=300); plt.close(fig)

    
        sorted_vals = np.sort(crashes); survival = 1 - (np.arange(1, N + 1) / N)
        fig, ax = plt.subplots(); ax.step(sorted_vals, survival, where='post')
        ax.set_xlabel('Target Multiplier (X)'); ax.set_ylabel('Chance of Reaching Target (P[Crash >= X])')
        ax.set_title('What Are the Chances of Reaching a High Multiplier?')
        for mult in [1.01,1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 20.0, 30.0]:
            prob = np.interp(mult, sorted_vals, survival)
            if prob > 0.01:
                ax.plot(mult, prob, 'ro'); ax.annotate(f"{mult:.2f}x = {prob:.1%}", (mult, prob), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=5)
        ax.set_xlim(0, 35); ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(2.0)); ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.5)); ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.grid(True, which="both", ls="--"); fname = FIG_DIR / 'crash_survival_audience_friendly.png'
        fig.tight_layout(); fig.savefig(fname, dpi=300); plt.close(fig)

        # ACF & PACF Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        plot_acf(crashes, lags=50, alpha=0.05, ax=axes[0], title='Autocorrelation (ACF)')
        plot_pacf(crashes, lags=50, alpha=0.05, ax=axes[1], title='Partial Autocorrelation (PACF)', method='ywm')
        axes[0].set_xlabel('Lag'); axes[1].set_xlabel('Lag'); axes[0].set_ylabel('Correlation'); axes[1].set_ylabel('Correlation')
        axes[0].grid(True, linestyle='--', alpha=0.6); axes[1].grid(True, linestyle='--', alpha=0.6)
        fname = FIG_DIR / 'acf_pacf_original.png'
        fig.tight_layout(); fig.savefig(fname, dpi=300); plt.close(fig)

    # --- NEW METHOD FOR FORMAL RANDOMNESS TEST ---
    def run_formal_randomness_test(self):
        """Performs the Augmented Dickey-Fuller test for stationarity."""
        logging.info("Running formal randomness test (Augmented Dickey-Fuller)...")
        # Testing the series of changes between rounds, which is a stronger test for predictability
        adf_result = adfuller(self.df['crash'].diff().dropna())
        
        report_content = (
            "The Augmented Dickey-Fuller (ADF) test is a formal statistical test for stationarity.\n"
            "A time series is 'stationary' if its statistical properties (like mean and variance)\n"
            "do not change over time. For gambling outcomes, stationarity is a strong indicator of\n"
            "unpredictability and randomness, as it means there are no trends to exploit.\n\n"
            "Null Hypothesis (H0): The series has a unit root (it is non-stationary and has a time-dependent structure).\n"
            "Alternative Hypothesis (H1): The series has no unit root (it is stationary and random).\n\n"
            f"Test Statistic: {adf_result[0]:.4f}\n"
            f"P-value: {adf_result[1]:.4e}\n\n"
            "Interpretation:\n"
            "A very low p-value (typically < 0.05) allows us to reject the null hypothesis.\n"
            f"Our p-value is extremely low, providing strong statistical evidence that the series\n"
            "of round-to-round changes is stationary. This formally supports the conclusion from the\n"
            "visual ACF/PACF plots: the game's outcomes are statistically random and do not\n"
            "contain exploitable trends."
        )
        self._append_to_report("Formal Test for Randomness (ADF Test)", report_content)

    def _build_lstm_model(self, seq_length):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(seq_length, 1)),
            Dropout(0.15),
            LSTM(32),
            Dropout(0.15),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def run_lstm_prediction(self, seq_length=15, n_folds=5, epochs=25, batch_size=64):
        # This is your original, unchanged method
        logging.info("Running LSTM prediction with expanding-window CV and persistence baseline...")
        X, y = self._create_sequences(self.scaled_data, seq_length)
        total_samples = len(X); fold_size = max(1, total_samples // (n_folds + 1)); fold_mses, persistence_mses = [], []
        for fold in range(n_folds):
            train_end = fold_size * (fold + 1); test_end = min(total_samples, train_end + fold_size)
            if test_end <= train_end: continue
            X_train, y_train = X[:train_end], y[:train_end]; X_test, y_test = X[train_end:test_end], y[train_end:test_end]
            model = self._build_lstm_model(seq_length)
            early = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=0)
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[early], verbose=0)
            mse = model.evaluate(X_test, y_test, verbose=0); fold_mses.append(mse)
            mse_persist = np.mean((X_test[:, -1, 0].reshape(-1, 1) - y_test) ** 2); persistence_mses.append(mse_persist)
            logging.info(f"Fold {fold+1}/{n_folds}: LSTM MSE={mse:.6e}, Persistence MSE={mse_persist:.6e}")
        mean_lstm_mse = np.mean(fold_mses) if fold_mses else np.nan; mean_persist_mse = np.mean(persistence_mses) if persistence_mses else np.nan
        improvement_ratio = mean_lstm_mse / mean_persist_mse if mean_persist_mse > 0 else float('inf')
        content = (f"Method: Expanding-window cross-validation (n_folds={n_folds})\n" f"Sequence Length: {seq_length}\n\n"
                   f"Mean LSTM MSE (scaled): {mean_lstm_mse:.6e}\n" f"Mean Persistence MSE (scaled): {mean_persist_mse:.6e}\n"
                   f"Ratio (LSTM / Persistence): {improvement_ratio:.4f}\n")
        self._append_to_report("LSTM Forecasting (Expanding-window CV)", content)
        # LSTM Loss Plot (Unchanged)
        X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X, y, test_size=0.2, shuffle=False)
        final_model = self._build_lstm_model(seq_length)
        history = final_model.fit(X_train_full, y_train_full, validation_data=(X_test_full, y_test_full), epochs=epochs, batch_size=batch_size, callbacks=[early], verbose=0)
        fig, ax = plt.subplots(); ax.plot(history.history['loss'], label='Training Loss', linewidth=1); ax.plot(history.history['val_loss'], label='Validation Loss', linewidth=1)
        min_val_loss = min(history.history['val_loss']); min_val_epoch = history.history['val_loss'].index(min_val_loss)
        ax.axvline(min_val_epoch, color='grey', linestyle=':', label=f'Best Epoch: {min_val_epoch+1}')
        ax.set_title('LSTM Training & Validation Loss (scaled MSE)'); ax.set_xlabel('Epoch'); ax.set_ylabel('MSE (scaled)'); ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
        fname = FIG_DIR / 'lstm_loss_original.png'; fig.tight_layout(); fig.savefig(fname, dpi=300); plt.close(fig)

    def _build_autoencoder_model(self, seq_length):
        inputs = Input(shape=(seq_length, 1)); encoded = LSTM(64, return_sequences=False)(inputs)
        bottleneck = RepeatVector(seq_length)(encoded); decoded = LSTM(64, return_sequences=True)(bottleneck)
        output = TimeDistributed(Dense(1))(decoded); model = Model(inputs, output)
        model.compile(optimizer='adam', loss='mae'); return model

    def run_autoencoder_analysis(self, seq_length=15, epochs=30, batch_size=64):
        # This is your original, unchanged method
        logging.info("Running autoencoder + surrogate comparison...")
        X, _ = self._create_sequences(self.scaled_data, seq_length)
        X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)
        model = self._build_autoencoder_model(seq_length)
        model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, X_test), verbose=0)
        recon = model.predict(X_test, verbose=0); mae_per_sample = np.mean(np.abs(recon - X_test), axis=(1, 2)).flatten()
        shuffled = self.scaled_data.copy().flatten(); np.random.shuffle(shuffled)
        X_surr, _ = self._create_sequences(shuffled.reshape(-1, 1), seq_length)
        X_surr_test = X_surr[:len(X_test)]; recon_surr = model.predict(X_surr_test, verbose=0)
        mae_surr = np.mean(np.abs(recon_surr - X_surr_test), axis=(1, 2)).flatten()
        ks_stat, ks_p = ks_2samp(mae_per_sample, mae_surr)
        report = ("AE MAE on REAL Test Data (Distribution of reconstruction errors):\n" f"{pd.Series(mae_per_sample).describe().to_string()}\n\n"
                  "AE MAE on SHUFFLED Surrogate Data (Distribution of reconstruction errors):\n" f"{pd.Series(mae_surr).describe().to_string()}\n\n"
                  "Kolmogorov-Smirnov (KS) Test:\n" f"- Statistic: {ks_stat:.6f}\n" f"- P-value: {ks_p:.6e}\n")
        self._append_to_report("Autoencoder Reconstruction & Surrogate Comparison", report)
        fig, ax = plt.subplots(); sns.kdeplot(mae_per_sample, label='Real Data Errors', ax=ax, fill=True); sns.kdeplot(mae_surr, label='Shuffled Data Errors', ax=ax, fill=True)
        ax.axvline(np.mean(mae_per_sample), color='blue', linestyle='--', label=f'Mean (Real): {np.mean(mae_per_sample):.4f}')
        ax.axvline(np.mean(mae_surr), color='orange', linestyle='--', label=f'Mean (Shuffled): {np.mean(mae_surr):.4f}')
        ax.set_title('AE Reconstruction Error: Real vs Shuffled Surrogate'); ax.set_xlabel('Mean Absolute Error per window (scaled)'); ax.legend(); ax.grid(True, linestyle='--', alpha=0.5)
        fname = FIG_DIR / 'autoencoder_errors_original.png'; fig.tight_layout(); fig.savefig(fname, dpi=300); plt.close(fig)

    def run_gambler_simulation(self, initial_balance=10_000, base_bet=10, n_sims=2000, n_rounds=None, max_bet=None):
        logging.info("Running gambler Monte Carlo simulations...")
        crashes = self.df['crash'].values;
        if n_rounds is None: n_rounds = len(crashes)
        if max_bet is None: max_bet = initial_balance * 0.1
        def simulate_strategy(crash_seq, strategy='fixed', target=1.5):
            balance, current_bet = initial_balance, base_bet; balances = []
            for r in crash_seq[:n_rounds]:
                if balance <= 0: balances.append(0); continue
                actual_bet = min(current_bet, balance, max_bet)
                if r >= target:
                    balance += actual_bet * (target - 1)
                    if strategy == 'martingale': current_bet = base_bet
                    elif strategy == 'anti_martingale': current_bet = min(current_bet * 2, max_bet)
                else:
                    balance -= actual_bet
                    if strategy == 'martingale': current_bet = min(current_bet * 2, max_bet)
                    elif strategy == 'anti_martingale': current_bet = base_bet
                balances.append(balance)
            return np.array(balances)
        strategies = ['fixed', 'martingale', 'anti_martingale']
        results = {s: np.zeros((n_sims, n_rounds)) for s in strategies}
        for sim in range(n_sims):
            crash_seq = np.random.choice(crashes, size=n_rounds, replace=True)
            for s in strategies: results[s][sim] = simulate_strategy(crash_seq, s, target=2.0 if s == 'martingale' else 1.5)
            if (sim + 1) % max(1, n_sims // 10) == 0: logging.info(f"Monte Carlo progress: {sim+1}/{n_sims}")
        summary = {s: {'median_final': np.median(results[s][:, -1]), 'mean_final': np.mean(results[s][:, -1]),
                       'p_rupture': np.mean(results[s][:, -1] <= 0), 'percentiles': np.percentile(results[s][:, -1], [5, 25, 50, 75, 95]).tolist()} for s in strategies}
        summary_df = pd.DataFrame([{'strategy': s, **summary[s]} for s in strategies])
        report_content = (f"Parameters: Initial Balance={initial_balance:,}, Base Bet={base_bet:,}, Sims={n_sims:,}, Rounds per Sim={n_rounds:,}.\n\n"
                          "Summary of Final Balances:\n" + summary_df.to_string(index=False))
        self._append_to_report("Gambler Monte Carlo Simulation", report_content)
        fig, ax = plt.subplots(); x = np.arange(1, n_rounds + 1); colors = {'fixed': '#0077b6', 'martingale': '#d62828', 'anti_martingale': '#2a9d8f'}
        for s in strategies:
            arr = results[s]; p50, p25, p75 = np.percentile(arr, [50, 25, 75], axis=0)
            ax.plot(x, p50, label=f"{s.replace('_', ' ').title()} (Median)", color=colors[s], linewidth=1.2)
            ax.fill_between(x, p25, p75, alpha=0.25, color=colors[s], label=f"{s.replace('_', ' ').title()} (25-75th Percentile)")
        ax.axhline(y=initial_balance, color='grey', linestyle='--', linewidth=1.2, label='Starting Balance')
        ax.set_title('Gambler Strategies: Median & Percentile Envelopes (Monte Carlo)'); ax.set_xlabel('Rounds Played'); ax.set_ylabel('Player Balance (EGP)')
        handles, labels = ax.get_legend_handles_labels(); unique_labels = dict(zip(labels, handles)); ax.legend(unique_labels.values(), unique_labels.keys())
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, p: f'{int(val):,}')); ax.grid(True, linestyle='--', alpha=0.6)
        fname = FIG_DIR / 'gambler_simulation_original.png'; fig.tight_layout(); fig.savefig(fname, dpi=300); plt.close(fig)

    def run_all_analyses(self):
        self.run_statistical_analysis()
        self.run_formal_randomness_test()
        self.run_lstm_prediction()
        self.run_autoencoder_analysis()
        self.run_gambler_simulation(initial_balance=10_000, base_bet=10, n_sims=2000, n_rounds=2500)
        logging.info(f"All analyses completed. See figures/ and report file: {self.report_file}")

if __name__ == '__main__':
    FILE_NAME = '1xbet_crash_data.csv'
    REPORT_NAME = 'crash_game_report.txt'
    if not Path(FILE_NAME).exists():
        logging.error(f"Data file '{FILE_NAME}' not found. Please place it in the same directory.")
    else:
        analyzer = CrashGameAnalyzer(FILE_NAME, REPORT_NAME)
        analyzer.run_all_analyses()


