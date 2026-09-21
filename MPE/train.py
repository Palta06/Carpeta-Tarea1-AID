import numpy as np
import pandas as pd
import os
import utility as ut # Importamos el script de utilidades que crearemos después

def sigmoid(z):
    """Función de activación Sigmoide."""
    # np.clip evita desbordamientos matemáticos (overflow) si z es muy grande o muy pequeño
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

def compute_loss(y, y_hat, weights, lambda_pen=0.0, penalized=False):
    """Calcula la Entropía Cruzada (Cross-Entropy) normal o penalizada."""
    m = len(y)
    epsilon = 1e-15 # Evita el log(0)
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    
    # Entropía Cruzada normal
    loss = -(1/m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    
    # Entropía Cruzada Penalizada (Regularización L2)
    if penalized:
        loss += (lambda_pen / (2 * m)) * np.sum(weights**2)
        
    return loss

def train_logistic_regression_mGD(X, y, epochs=1000, lr=0.1, beta=0.9, penalized=False, lambda_pen=0.1):
    """
    Entrena el modelo usando Descenso del Gradiente con Momentum (mGD).
    """
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0.0
    
    # Inicialización de las velocidades para el momentum
    v_w = np.zeros(n)
    v_b = 0.0
    
    loss_history = []
    
    for i in range(epochs):
        # Forward pass (Predicción)
        z = np.dot(X, weights) + bias
        y_hat = sigmoid(z)
        
        # Calcular pérdida (Entropía Cruzada) y guardar convergencia
        loss = compute_loss(y, y_hat, weights, lambda_pen, penalized)
        loss_history.append(loss)
        
        # Backward pass (Cálculo de Gradientes)
        dw = (1/m) * np.dot(X.T, (y_hat - y))
        db = (1/m) * np.sum(y_hat - y)
        
        # Añadir la penalización a los gradientes si corresponde
        if penalized:
            dw += (lambda_pen / m) * weights
            
        # Actualización de mGD (Momentum Gradient Descent)
        v_w = beta * v_w + (1 - beta) * dw
        v_b = beta * v_b + (1 - beta) * db
        
        # Actualizar los pesos y el sesgo (bias)
        weights -= lr * v_w
        bias -= lr * v_b
        
    return weights, bias, loss_history

def save_csv_results(filename, data, columns):
    """Guarda los resultados numéricos en formato CSV."""
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)
    print(f"Archivo guardado: {filename}")

if __name__ == "__main__":
    print("--- Iniciando Entrenamiento de Regresión Logística (MPE) ---")
    
    # 1. Cargar las características (MPE) desde utility.py
    # Nota: Descomentaremos esto cuando utility.py esté listo.
    # X_train, y_train, X_test, y_test = ut.get_mpe_features()
    
    # MOCK DATA: Datos simulados temporalmente para probar que el script funciona
    # BORRAR ESTO cuando conectemos con utility.py
    np.random.seed(42)
    X_train = np.random.rand(100, 5) # 100 ventanas, 5 escalas de entropía
    y_train = np.random.randint(0, 2, 100) # Etiquetas 0 (Normal) y 1 (Fallo)
    # -----------------------------------------------------------
    
    # Hiperparámetros del modelo
    EPOCHS = 1500
    LEARNING_RATE = 0.05
    MOMENTUM_BETA = 0.9
    LAMBDA_PENALTY = 0.1
    
    # 2. Entrenamiento Normal (Entropía Cruzada Estándar)
    print("\nEntrenando modelo Normal...")
    weights_norm, bias_norm, loss_hist_norm = train_logistic_regression_mGD(
        X_train, y_train, epochs=EPOCHS, lr=LEARNING_RATE, beta=MOMENTUM_BETA, penalized=False
    )
    
    # 3. Entrenamiento Penalizado (Entropía Cruzada Penalizada)
    print("Entrenando modelo Penalizado...")
    weights_pen, bias_pen, loss_hist_pen = train_logistic_regression_mGD(
        X_train, y_train, epochs=EPOCHS, lr=LEARNING_RATE, beta=MOMENTUM_BETA, penalized=True, lambda_pen=LAMBDA_PENALTY
    )
    
    # 4. Exportar Entregables Numéricos (CSV)
    print("\nExportando resultados a CSV...")
    
    # Guardar Curvas de Convergencia
    convergence_data = {
        'Epoch': np.arange(1, EPOCHS + 1),
        'Loss_Normal': loss_hist_norm,
        'Loss_Penalizada': loss_hist_pen
    }
    save_csv_results('mpe_convergencia_mGD.csv', convergence_data, ['Epoch', 'Loss_Normal', 'Loss_Penalizada'])
    
    # Guardar Coeficientes de Regresión
    # Añadimos el bias (intercepto) como el primer coeficiente (W0)
    coef_data = {
        'Coeficiente': ['W0 (Bias)'] + [f'W{i}' for i in range(1, len(weights_norm) + 1)],
        'Valor_Normal': [bias_norm] + list(weights_norm),
        'Valor_Penalizado': [bias_pen] + list(weights_pen)
    }
    save_csv_results('mpe_coeficientes_regresion.csv', coef_data, ['Coeficiente', 'Valor_Normal', 'Valor_Penalizado'])
    
    print("\n¡Entrenamiento completado y entregables guardados!")