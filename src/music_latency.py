import numpy as np
from scipy.linalg import eigh

class MUSICDelayEstimator:
    """
    MUSIC algorithm for estimating time delays between correlated signals.
    
    Given an array of N assets, we model each as:
        x_i(t) = a_i * s(t - τ_i) + noise
    The algorithm computes the covariance matrix of the observations,
    separates signal and noise subspaces, and finds peaks in the
    pseudo‑spectrum corresponding to the delays.
    """
    def __init__(self, n_sources=1):
        """
        n_sources : number of distinct delays (usually 1 for lead‑lag)
        """
        self.n_sources = n_sources

    def fit(self, X):
        """
        X : numpy array of shape (n_samples, n_assets)
        Builds the covariance matrix and finds the noise subspace.
        """
        # Center the data (remove mean)
        X_centered = X - X.mean(axis=0)
        # Covariance matrix (sample covariance)
        R = X_centered.T @ X_centered / (X.shape[0] - 1)
        
        # Eigenvalue decomposition
        eigvals, eigvecs = eigh(R)
        # Sort in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Noise subspace: eigenvectors corresponding to smallest eigenvalues
        self.noise_subspace = eigvecs[:, self.n_sources:]
        return self

    def compute_spectrum(self, delays, fs=1.0, f0=1.0):
        """
        Compute the MUSIC pseudo‑spectrum over a range of delays (in seconds).
        delays : array of candidate delays (e.g., np.linspace(-0.1, 0.1, 1000))
        fs     : sampling frequency (samples per second)
        f0     : reference frequency (Hz) – used for narrowband steering
        Returns array of spectrum values (higher peak = better match).
        """
        n_assets = self.noise_subspace.shape[0]
        spectrum = []
        for tau in delays:
            # Steering vector: first asset reference (delay 0), others delay tau
            steering = np.ones(n_assets, dtype=complex)
            for i in range(1, n_assets):
                steering[i] = np.exp(-1j * 2 * np.pi * f0 * tau)
            # Project onto noise subspace
            proj = np.linalg.norm(self.noise_subspace.conj().T @ steering)
            spectrum.append(1.0 / (proj + 1e-12))
        return np.array(spectrum)
