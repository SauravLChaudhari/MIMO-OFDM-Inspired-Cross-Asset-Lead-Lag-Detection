import matplotlib.pyplot as plt
import numpy as np

def plot_cross_correlation(returns, symbols, max_lag=50):
    """
    Plot pairwise cross‑correlation to see lead‑lag visually.
    """
    n = len(symbols)
    fig, axes = plt.subplots(n, n, figsize=(12, 12))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(returns.iloc[:, i], bins=30, alpha=0.5)
                ax.set_title(f'{symbols[i]}')
            else:
                lags = np.arange(-max_lag, max_lag+1)
                corr = [returns.iloc[:, i].shift(lag).corr(returns.iloc[:, j]) for lag in lags]
                ax.plot(lags, corr)
                ax.axhline(0, color='black', linestyle='--')
                ax.axvline(0, color='red', linestyle='--')
            if i == n-1:
                ax.set_xlabel('Lag')
            if j == 0:
                ax.set_ylabel(symbols[i])
    plt.tight_layout()
    plt.show()

def plot_music_spectrum(delays, spectrum, true_delay=None):
    """
    Plot MUSIC pseudo‑spectrum.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(delays, spectrum)
    if true_delay is not None:
        plt.axvline(true_delay, color='red', linestyle='--', label='True delay')
    plt.xlabel('Delay (seconds)')
    plt.ylabel('MUSIC Spectrum')
    plt.title('MUSIC Pseudo‑Spectrum for Lead‑Lag Detection')
    plt.legend()
    plt.grid(True)
    plt.show()
