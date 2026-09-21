import os
import numpy as np
import scipy.io as sio
from scipy.signal import decimate
from scipy.stats import norm

def get_de_time(file_path):
    mat = sio.loadmat(file_path)
    for key in mat.keys():
        if 'DE_time' in key: return mat[key].flatten()
    return None

def coarse_graining(signal, scale):
    n = len(signal)
    max_len = n - (n % scale)
    return np.mean(signal[:max_len].reshape(-1, scale), axis=1)

def dispersion_entropy(signal, classes=6, m=2, delay=1):
    n = len(signal)
    if n < m * delay: return 0.0
    mu, sigma = np.mean(signal), np.std(signal)
    if sigma == 0: return 0.0
    
    y = norm.cdf(signal, loc=mu, scale=sigma)
    z = np.clip(np.round(classes * y + 0.5), 1, classes).astype(int)
    
    idx = np.arange(n - (m - 1) * delay)
    indices = idx[:, None] + np.arange(m) * delay
    patterns = z[indices]
    
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probs = counts / np.sum(counts)
    return -np.sum(probs * np.log2(probs + 1e-10)) / np.log2(classes**m)

def multiscale_de(signal, classes=6, m=2, delay=1, max_scale=10):
    return np.array([dispersion_entropy(coarse_graining(signal, s), classes, m, delay) for s in range(1, max_scale + 1)])

def get_features(window_size=1200, overlap=0.5, test_size=0.3):
    X_tr, y_tr, X_ts, y_ts = [], [], [], []
    base_path = os.path.join(os.path.dirname(__file__), '..')
    carpetas = [(os.path.join(base_path, 'datas_normal'), 0), 
                (os.path.join(base_path, 'datas_fallo'), 1)]
    
    step = int(window_size * (1 - overlap))
    archivos_48k = {'97.mat', '98.mat', '99.mat', '100.mat'}

    for carpeta, label in carpetas:
        if not os.path.exists(carpeta): continue
        for filename in os.listdir(carpeta):
            if not filename.endswith('.mat'): continue
            signal = get_de_time(os.path.join(carpeta, filename))
            if signal is None: continue
            
            if filename in archivos_48k or label == 0:
                signal = decimate(signal, 4)
                
            split_idx = int(len(signal) * (1 - test_size))
            sig_train, sig_test = signal[:split_idx], signal[split_idx:]
            
            for sig, X_list, y_list in [(sig_train, X_tr, y_tr), (sig_test, X_ts, y_ts)]:
                for i in range(0, len(sig) - window_size + 1, step):
                    window = sig[i : i + window_size]
                    X_list.append(multiscale_de(window, classes=6, m=2, delay=1, max_scale=10))
                    y_list.append(label)
                    
    return np.array(X_tr), np.array(X_ts), np.array(y_tr), np.array(y_ts)