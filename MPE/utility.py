import os
import numpy as np
import scipy.io as sio
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
    max_len = n - (n % scale) # Cortar el final para que sea divisible exacto por la escala
    return np.mean(signal[:max_len].reshape(-1, scale), axis=1)

def permutation_entropy(signal, m=3, delay=1):
    """Calcula la Entropía de Permutación (PE) normalizada."""
    n = len(signal)
    if n < m * delay:
        return 0.0
    
    # Extraer patrones ordinales
    patterns = []
    for i in range(n - (m - 1) * delay):
        segment = signal[i : i + m * delay : delay]
        patterns.append(tuple(np.argsort(segment)))
    
    # Calcular probabilidades de cada patrón
    counts = Counter(patterns)
    probs = np.array(list(counts.values())) / len(patterns)
    
    # Entropía de Shannon Normalizada
    m_factorial = np.math.factorial(m)
    pe = -np.sum(probs * np.log2(probs + 1e-10)) / np.log2(m_factorial)
    return pe

def multiscale_permutation_entropy(signal, m=3, delay=1, max_scale=5):
    """Calcula la MPE obteniendo la entropía para múltiples escalas."""
    mpe_vals = []
    for scale in range(1, max_scale + 1):
        cg_signal = coarse_graining(signal, scale)
        pe = permutation_entropy(cg_signal, m, delay)
        mpe_vals.append(pe)
    return np.array(mpe_vals)

def get_mpe_features(data_dir='../Datos', window_size=1200, test_size=0.3):
    """
    Función principal que orquesta la carga, partición y cálculo de MPE.
    Devuelve los conjuntos X_train, X_test, y_train, y_test listos para el modelo.
    """
    X = []
    y = []
    
    # Si se ejecuta desde la raíz de la carpeta en lugar de dentro de MPE/
    if not os.path.exists(data_dir):
        data_dir = 'Datos'
        
    print(f"Cargando archivos .mat desde la carpeta: {data_dir}")
    
    archivos = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
    
    for filename in archivos:
        filepath = os.path.join(data_dir, filename)
        signal = get_de_time(filepath)
        
        if signal is None:
            continue
            
        # Asignar la etiqueta: 0 si el nombre contiene "Normal", 1 si es falla
        label = 0 if 'Normal' in filename else 1
        
        # Particionar la señal en ventanas de tamaño W (window_size)
        n_windows = len(signal) // window_size
        for i in range(n_windows):
            window = signal[i * window_size : (i + 1) * window_size]
            features = multiscale_permutation_entropy(window, m=3, delay=1, max_scale=5)
            X.append(features)
            y.append(label)
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"Procesamiento terminado. Ventanas totales: {len(X)}")
    
    # Stratify=y asegura que la proporción de datos normales y de falla sea igual en train y test
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)