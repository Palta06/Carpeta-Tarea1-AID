import os
import math
import numpy as np
import scipy.io as sio
from collections import Counter
from sklearn.model_selection import train_test_split

def get_de_time(file_path):
    mat = sio.loadmat(file_path)
    for key in mat.keys():
        if 'DE_time' in key:
            return mat[key].flatten()
    return None

def coarse_graining(signal, scale):
    n = len(signal)
    max_len = n - (n % scale)
    return np.mean(signal[:max_len].reshape(-1, scale), axis=1)

def permutation_entropy(signal, m=3, delay=1):
    n = len(signal)
    if n < m * delay: return 0.0
    patterns = []
    for i in range(n - (m - 1) * delay):
        segment = signal[i : i + m * delay : delay]
        patterns.append(tuple(np.argsort(segment)))
    counts = Counter(patterns)
    probs = np.array(list(counts.values())) / len(patterns)
    m_factorial = math.factorial(m)
    return -np.sum(probs * np.log2(probs + 1e-10)) / np.log2(m_factorial)

def multiscale_permutation_entropy(signal, m=3, delay=1, max_scale=5):
    return np.array([permutation_entropy(coarse_graining(signal, s), m, delay) for s in range(1, max_scale + 1)])

def get_mpe_features(window_size=1200, test_size=0.3):
    X, y = [], []
    
    base_path = "." if os.path.exists("datas_normal") else ".."
    carpetas = [(os.path.join(base_path, 'datas_normal'), 0), 
                (os.path.join(base_path, 'datas_fallo'), 1)]
    
    for carpeta, label in carpetas:
        print(f"Cargando datos desde: {carpeta}")
        if not os.path.exists(carpeta): 
            print(f"Advertencia: No se encontró {carpeta}")
            continue
            
        for filename in os.listdir(carpeta):
            if not filename.endswith('.mat'): continue
            signal = get_de_time(os.path.join(carpeta, filename))
            if signal is None: continue
            
            n_windows = len(signal) // window_size
            for i in range(n_windows):
                window = signal[i * window_size : (i + 1) * window_size]
                X.append(multiscale_permutation_entropy(window, m=3, delay=1, max_scale=5))
                y.append(label)
                
    X, y = np.array(X), np.array(y)
    print(f"Procesamiento MPE terminado. Ventanas extraídas: {len(X)}")
    
    # Partición estratificada
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Estandarización Z-score (Media 0, Desviación Estándar 1) basada estrictamente en Train
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0 # Evitar división por cero
    
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    return X_train, X_test, y_train, y_test