import numpy as np
import pandas as pd
import json
import os
import utility as ut

def sigmoid(z): 
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

def compute_loss(y, y_hat, weights, lambda_pen=0.0, penalized=False):
    m = len(y)
    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
    loss = -(1/m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    if penalized: loss += (lambda_pen / (2 * m)) * np.sum(weights**2)
    return loss

def train_logistic_regression_mGD(X, y, epochs=1000, lr=0.1, beta=0.9, penalized=False, lambda_pen=0.1):
    m, n = X.shape
    weights, bias = np.zeros(n), 0.0
    v_w, v_b = np.zeros(n), 0.0
    loss_history = []
    
    for _ in range(epochs):
        y_hat = sigmoid(np.dot(X, weights) + bias)
        loss_history.append(compute_loss(y, y_hat, weights, lambda_pen, penalized))
        
        dw = (1/m) * np.dot(X.T, (y_hat - y))
        db = (1/m) * np.sum(y_hat - y)
        if penalized: dw += (lambda_pen / m) * weights
            
        v_w = beta * v_w + (1 - beta) * dw
        v_b = beta * v_b + (1 - beta) * db
        weights -= lr * v_w
        bias -= lr * v_b
        
    return weights, bias, loss_history

if __name__ == "__main__":
    print("--- Iniciando Entrenamiento MPE ---")
    
    # 1. Definir hiperparámetros (Se exportarán al JSON de tu compañero)
    params = {
        "window_size": 1200,
        "epochs": 1500,
        "learning_rate": 0.05,
        "momentum_beta": 0.9,
        "lambda_penalty": 0.1,
        "optimizador": "mGD (Descenso de Gradiente con Momentum)",
        "extraccion": "Entropía Multi-escala de Permutación (MPE: m=3, delay=1, max_scale=5)"
    }
    
    # Extraer datos reales (sin Mock Data)
    X_train, X_test, y_train, y_test = ut.get_mpe_features(window_size=params["window_size"])
    
    # Asegurar que se guarde dentro de MPE/ sin importar desde dónde se ejecute
    out_dir = "MPE" if os.path.exists("MPE") else "."
    
    # 2. Guardar parametros.json
    with open(os.path.join(out_dir, 'parametros.json'), 'w') as f:
        json.dump(params, f, indent=4)
        
    # 3. Entrenar modelos
    print("\nEntrenando modelo Normal...")
    _, _, loss_norm = train_logistic_regression_mGD(
        X_train, y_train, epochs=params["epochs"], lr=params["learning_rate"], beta=params["momentum_beta"], penalized=False
    )
    
    print("Entrenando modelo Penalizado (Regularización L2)...")
    weights_pen, bias_pen, loss_pen = train_logistic_regression_mGD(
        X_train, y_train, epochs=params["epochs"], lr=params["learning_rate"], beta=params["momentum_beta"], 
        penalized=True, lambda_pen=params["lambda_penalty"]
    )
    
    # 4. Guardar pesos en .npz (Lógica del compañero)
    np.savez(os.path.join(out_dir, 'mejor_modelo.npz'), weights=weights_pen, bias=bias_pen)
    
    # 5. Guardar CSVs
    df_conv = pd.DataFrame({'Epoch': np.arange(1, params["epochs"] + 1), 'Loss_Normal': loss_norm, 'Loss_Penalizada': loss_pen})
    df_conv.to_csv(os.path.join(out_dir, 'mpe_convergencia_mGD.csv'), index=False)
    
    print(f"\n¡Archivos MPE (.npz, .json, .csv) guardados exitosamente dentro de la carpeta {out_dir}/!")