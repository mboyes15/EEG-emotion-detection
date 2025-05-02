import numpy as np
from scipy.fftpack import fft, ifft
from sklearn.cross_decomposition import CCA
from scipy.signal import welch

def robust_stats(signal):
    """Compute robust mean & scale using median and MAD."""
    median = np.median(signal)
    mad = np.median(np.abs(signal - median)) * 1.4826  # Convert MAD to std estimate
    return median, mad

# Variational Mode Decomposition (VMD)
def VMD(signal, alpha=400, tau=0.5, K=5, DC=0, init=1, tol=1e-6):
    signal = np.asarray(signal)  # Ensure it's a NumPy array
    N = len(signal)
#     print(f'length N: {N}')
    
    # Frequency domain
    freq = np.arange(1, N + 1) / N - 0.5 - (1 / N) * np.mod(N, 2)
    
    # Initialization
    u = np.zeros((K, N))  # Modes
    omega = initialize_omega_psd(signal, K)  # Use filtered peak frequencies
    omega += np.random.uniform(-0.05, 0.05, size=omega.shape)

#     np.linspace(0, 0.5, K)  # Center frequencies
    lambda_hat = np.zeros(N, dtype=complex)  # Lagrange multiplier (complex)
    
    # Transform to Fourier domain
    f_hat = fft(signal)
    
    u_hat = np.zeros((K, N), dtype=complex)
    
    iter_count = 0
    omega_old = np.copy(omega)
#     print("reached loop")
    while True:
        for k in range(K):
            sum_u_hat = np.sum(u_hat, axis=0) - u_hat[k]
            residual = f_hat - sum_u_hat - lambda_hat / 2
            
            # Update u_hat[k]
            u_hat[k] = residual / (1 + alpha * (freq - omega[k]) ** 2)
            
            # Update omega[k]
            omega[k] = np.sum(np.abs(freq) * np.abs(u_hat[k]) ** 2) / np.sum(np.abs(u_hat[k]) ** 2)
#         if iter_count % 10 == 0:
#             print(f"Iteration {iter_count}: omega = {omega}")

        # Update lambda_hat
        lambda_hat += tau * (np.sum(u_hat, axis=0) - f_hat)
#         print(f"lambda_hat: {lambda_hat}")
        # Check convergence
        if np.linalg.norm(omega - omega_old) < tol:
#             print("converged")
            break
        
        omega_old = np.copy(omega)
        iter_count += 1
    
    # Compute final modes
    for k in range(K):
        u[k] = np.real(ifft(u_hat[k]))
    
    return u

# Canonical Correlation Analysis (CCA) for Denoising
def CCA_denoise(modes, num_components=1):
    cca = CCA(n_components=num_components)
    X1, X2 = modes[:-1], modes[1:]
    cca.fit(X1.T, X2.T)
    cleaned_signal = cca.transform(X1.T, X2.T)[0].T
    return cleaned_signal



# Example usage
def denoise_eeg(signal):
    modes = VMD(signal, K=5)  # Decompose into 4 modes
#     for i, mode in enumerate(modes):
#         print(f"Mode {i}: mean={np.mean(mode)}, std={np.std(mode)}, max={np.max(mode)}, min={np.min(mode)}")
    for i in range(modes.shape[0]):
        modes[i] = (modes[i] - np.mean(modes[i])) / (np.std(modes[i]) + 1e-8)

    clean_signal = CCA_denoise(modes)  # Denoise using CCA
    print(f"Original signal: mean={np.mean(signal)}, std={np.std(signal)}")
#     print(f"Clean signal (before scaling): mean={np.mean(clean_signal)}, std={np.std(clean_signal)}, max={np.max(clean_signal)}, min={np.min(clean_signal)}")
    # Get robust statistics from noisy input
    estimated_mean, estimated_std = robust_stats(signal)


    
    # Scale using the estimated clean signal statistics
    clean_signal = (clean_signal- np.mean(clean_signal)) / (np.std(clean_signal) + 1e-8)
    clean_signal = clean_signal * estimated_std + estimated_mean

    print(f"Final Clean Signal: mean={np.mean(clean_signal)}, std={np.std(clean_signal)}")



    return clean_signal, modes

def filter_extreme_peaks(freqs, psd, threshold=95):
    """ Filter out extreme PSD peaks based on a threshold percentile. """
    cutoff = np.percentile(psd, threshold)  # Get the 95th percentile value
    valid_indices = psd <= cutoff  # Boolean mask for keeping values below the threshold
    
    # Ensure valid_indices is applied correctly
#     print(freqs)
    freqs_filtered = freqs[valid_indices]  # Apply mask to frequencies
    psd_filtered = psd[valid_indices]  # Apply mask to PSD values

    return freqs_filtered, psd_filtered

def initialize_omega_psd(signal, K):
    """
    Initialize omega using the dominant frequencies of the signal,
    with extreme peak filtering.
    """
    freqs, psd = welch(signal, nperseg=len(signal)//8)
    freqs, psd = filter_extreme_peaks(freqs, psd)
    
    # Select top-K dominant frequencies
    dominant_freqs = freqs[np.argsort(psd)[-K:]]
    return np.sort(dominant_freqs)

def hurst_exponent(signal):
    N = len(signal)
    mean_signal = np.mean(signal)
    Y = np.cumsum(signal - mean_signal)
    R = np.max(Y) - np.min(Y)
    S = np.std(signal)

    if S == 0:
        return 0.5  # Neutral value if no variability
    return np.log(R / S) / np.log(N)

def higuchi_fd(signal, kmax=10):
    N = len(signal)
    L = []

    for k in range(1, kmax + 1):
        Lk = []
        for m in range(k):
            idx = np.arange(1, int(np.floor((N - m) / k)), dtype=int)
            Lmk = np.sum(np.abs(signal[m + idx * k] - signal[m + k * (idx - 1)]))
            norm = (N - 1) / (len(idx) * k)
            Lmk = (Lmk * norm) / k
            Lk.append(Lmk)
        L.append(np.mean(Lk))

    lnL = np.log(L)
    lnk = np.log(1.0 / np.arange(1, kmax + 1))
    return np.polyfit(lnk, lnL, 1)[0]
def tsallis_entropy(signal, q=2.0, bins=100):
    hist, _ = np.histogram(signal, bins=bins, density=True)
    hist = hist[hist > 0]  # Avoid log(0)

    if q == 1:
        return -np.sum(hist * np.log(hist))  # Shannon entropy
    else:
        return (1 - np.sum(hist ** q)) / (q - 1)

def sample_entropy(signal, m=2, r=0.2):
    N = len(signal)
    r *= np.std(signal)

    def _phi(m):
        x = np.array([signal[i:i + m] for i in range(N - m + 1)])
        C = np.sum([np.sum(np.linalg.norm(x - x_i, axis=1) <= r) - 1 for x_i in x])
        return C / ((N - m + 1) * (N - m))

    return -np.log(_phi(m + 1) / _phi(m))
