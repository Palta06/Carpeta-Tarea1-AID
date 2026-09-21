import numpy as np
import pandas as pd
import json, os
import utility as ut
from sklearn.metrics import f1_score

def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

def train_mGD(X, y, epochs=1500, lr=0.5, beta=0.9, lambda_pen=0.0):
    m_samples, n_feats = X.shape
    w, b = np.zeros(n_feats), 0.0
    v_w, v_b = np.zeros(n_feats), 0.0
    loss_hist = []
    
    # Pesos de clase para combatir desbalance
    w_0 = m_samples / (2.0 * np.sum(y == 0)) if np.sum(y==0)>0 else 1.0
    w_1 = m_samples / (2.0 * np.sum(y == 1)) if np.sum(y==1)>0 else 1.0
    sw = np.where(y == 0, w_0, w_1)
    
    for _ in range(epochs):
        y_hat = sigmoid(np.dot(X, w) + b)
        y_hat_clip = np.clip(y_hat, 1e-15, 1 - 1e-15)
        # Fórmula L2 estándar (sin penalizar sesgo)
        loss = -(1/m_samples) * np.sum(sw * (y * np.log(y_hat_clip) + (1 - y) * np.log(1 - y_hat_clip)))
        loss += 0.5 * lambda_pen * np.sum(w**2)
        loss_hist.append(loss)
        
        error = sw * (y_hat - y)
        dw = (1/m_samples) * np.dot(X.T, error) + lambda_pen * w
        db = (1/m_samples) * np.sum(error)
        
        v_w = beta * v_w + (1 - beta) * dw
        v_b = beta * v_b + (1 - beta) * db
        w -= lr * v_w
        b -= lr * v_b
        
    return w, b, loss_hist

if __name__ == "__main__":
    out_dir = os.path.dirname(__file__)
    windows = [600, 1200, 2400]
    lambdas = [0.0, 0.001, 0.01, 0.1]
    best_f1, best_params, best_model = -1, {}, {}
    resumen = []
    
    print("Iniciando Grid Search MPE...")
    for W in windows:
        X_tr_raw, X_ts_raw, y_tr, y_ts = ut.get_features(window_size=W)
        mean, std = np.mean(X_tr_raw, axis=0), np.std(X_tr_raw, axis=0)
        std[std == 0] = 1.0
        X_tr, X_ts = (X_tr_raw - mean) / std, (X_ts_raw - mean) / std
        
        # Modelo Normal base para esta ventana
        w_n, b_n, loss_n = train_mGD(X_tr, y_tr, lambda_pen=0.0)
        
        for lam in lambdas:
            w_p, b_p, loss_p = train_mGD(X_tr, y_tr, lambda_pen=lam)
            y_pred = (sigmoid(np.dot(X_ts, w_p) + b_p) >= 0.5).astype(int)
            f1 = f1_score(y_ts, y_pred, average='macro')
            resumen.append({'W': W, 'Lambda': lam, 'F1_Macro_Test': f1})
            
            if f1 > best_f1:
                best_f1 = f1
                best_params = {"metodo": "MPE", "window_size": W, "lambda_penalty": lam, "learning_rate": 0.5, "epochs": 1500}
                best_model = {'w_p': w_p, 'b_p': b_p, 'loss_p': loss_p, 'w_n': w_n, 'b_n': b_n, 'loss_n': loss_n, 'mean': mean, 'std': std}

    # Guardar Entregables
    pd.DataFrame(resumen).to_csv(os.path.join(out_dir, 'resumen_modelos.csv'), index=False)
    np.savez(os.path.join(out_dir, 'mejor_modelo.npz'), **best_model)
    with open(os.path.join(out_dir, 'parametros.json'), 'w') as f: json.dump(best_params, f, indent=4)
    
    df_conv = pd.DataFrame({'Epoch': np.arange(1, 1501), 'Loss_Normal': best_model['loss_n'], 'Loss_Penalizada': best_model['loss_p']})
    df_conv.to_csv(os.path.join(out_dir, 'mpe_convergencia_mGD.csv'), index=False)
    
    df_coefs = pd.DataFrame({
        'Coeficiente': ['Bias'] + [f'Escala_{i+1}' for i in range(len(best_model['w_p']))],
        'Normal': [best_model['b_n']] + list(best_model['w_n']),
        'Penalizado': [best_model['b_p']] + list(best_model['w_p'])
    })
    df_coefs.to_csv(os.path.join(out_dir, 'mpe_coeficientes_regresion.csv'), index=False)
    print("MPE Train Completado.")