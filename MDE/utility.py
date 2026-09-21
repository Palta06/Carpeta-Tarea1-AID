import os
import numpy as np
import scipy.io as sio
from scipy.stats import norm
from collections import Counter
from sklearn.model_selection import train_test_split

def get_de_time(file_path):
    """Abre el archivo .mat y extrae dinámicamente la variable que contiene 'DE_time'."""
    mat = sio.loadmat(file_path)
    for key in mat.keys():
        if 'DE_time' in key:
            return mat[key].flatten()
    return None

def coarse_graining(signal, scale):
    """Aplica el procedimiento de 'coarse-graining' (promediado) para una escala específica."""
    n = len(signal)
    max_len = n - (n % scale) 
    return np.mean(signal[:max_len].reshape(-1, scale), axis=1)

def dispersion_entropy(signal, classes=3, m=2, delay=1):
    """Calcula la Entropía de Dispersión (DE) normalizada."""
    n = len(signal)
    if n < m * delay:
        return 0.0
    
    # 1. Mapeo usando la Función de Distribución Acumulada Normal (NCDF)
    mu, sigma = np.mean(signal), np.std(signal)
    if sigma == 0:
        return 0.0
    
    y = norm.cdf(signal, loc=mu, scale=sigma)
    
    # 2. Asignar los valores a clases discretas (de 1 a 'classes')
    z = np.round(classes * y + 0.5).astype(int)
    z = np.clip(z, 1, classes)
    
    # 3. Extraer los patrones de dispersión
    patterns = []
    for i in range(n - (m - 1) * delay):
        patterns.append(tuple(z[i : i + m * delay : delay]))
        
    # 4. Calcular la probabilidad de cada patrón
    counts = Counter(patterns)
    probs = np.array(list(counts.values())) / len(patterns)
    
    # 5. Calcular Entropía de Shannon Normalizada
    max_entropy = np.log2(classes**m)
    de = -np.sum(probs * np.log2(probs + 1e-10)) / max_entropy
    return de

def multiscale_dispersion_entropy(signal, classes=3, m=2, delay=1, max_scale=5):
    """Calcula la MDE obteniendo la entropía para múltiples escalas."""
    mde_vals = []
    for scale in range(1, max_scale + 1):
        cg_signal = coarse_graining(signal, scale)
        de = dispersion_entropy(cg_signal, classes, m, delay)
        mde_vals.append(de)
    return np.array(mde_vals)

def get_mde_features(data_dir='../Datos', window_size=1200, test_size=0.3):
    """
    Función principal que orquesta la carga, partición y cálculo de MDE.
    Devuelve los conjuntos listos para alimentar la Regresión Logística.
    """
    X = []
    y = []
    
    if not os.path.exists(data_dir):
        data_dir = 'Datos'
        
    print(f"Cargando archivos .mat desde la carpeta: {data_dir} para calcular MDE")
    
    archivos = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
    
    for filename in archivos:
        filepath = os.path.join(data_dir, filename)
        signal = get_de_time(filepath)
        
        if signal is None:
            continue
            
        # Asignar la etiqueta: 0 si es Normal, 1 si es falla
        label = 0 if 'Normal' in filename else 1
        
        # Particionar la señal en ventanas de tamaño W
        n_windows = len(signal) // window_size
        for i in range(n_windows):
            window = signal[i * window_size : (i + 1) * window_size]
            features = multiscale_dispersion_entropy(window, classes=3, m=2, delay=1, max_scale=5)
            X.append(features)
            y.append(label)
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"Procesamiento MDE terminado. Ventanas totales: {len(X)}")
    
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)