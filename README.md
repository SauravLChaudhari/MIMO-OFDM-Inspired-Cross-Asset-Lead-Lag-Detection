# MIMO‑OFDM Inspired Cross‑Asset Lead‑Lag Detection

## Signal‑Processing to Quant Translation

In MIMO‑OFDM (Wi‑Fi), the **Channel State Information (CSI)** captures how signals travel through multiple paths (multipath).  
Each path has a **time of flight** (delay) and **phase shift**.  
Algorithms like **MUSIC** (MUltiple SIgnal Classification) resolve these delays with sub‑sample precision, even when the sampling rate is lower than the delay.

In finance, when news affects correlated assets, the information propagates from one asset to another – exactly like a signal propagating through a multipath channel.  
The **lead‑lag relationship** is the delay between assets.

Here, I treat a basket of correlated assets (e.g., SPY, QQQ, AAPL, MSFT) as an **antenna array**.  
The returns over time form the “received signal”.  
I apply **MUSIC** to estimate the **propagation delays** between assets – i.e., which asset leads and by how many milliseconds (or samples).

## Mathematical Model

Let `y_i(t)` be the log‑price of asset `i` at time `t`.  
We assume a single “source” signal (the news) that reaches each asset with a delay `τ_i` and an amplitude `a_i`:

`y_i(t) = a_i * s(t - τ_i) + noise`

In vector form:  
`y(t) = a * s(t - τ) + noise`

MUSIC constructs a **covariance matrix** of the returns and performs an eigenvalue decomposition.  
The eigenvectors span a **signal subspace** (the delays) and a **noise subspace**.  
By projecting a steering vector onto the noise subspace, we find peaks at the true delays.

This is the same mathematics used to resolve multipath in wireless channels.

## Repository Contents

- `src/` – core modules
  - `music_latency.py`: MUSIC implementation for delay estimation
  - `data_loader.py`: fetches asset data (Yahoo Finance or Kaggle)
  - `visualization.py`: plots cross‑correlations and MUSIC spectra
- `notebooks/` – interactive demo
- `tests/` – unit tests

## Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
