import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import plot
import utility as ut
import os

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

def predict(X, weights, bias, threshold=0.5):
    z = np.dot(X, weights) + bias
    return (sigmoid(z) >= threshold).astype(int)

if __name__ == "__main__":
    print("--- Iniciando Evaluación MPE ---")
    out_dir = "MPE" if os.path.exists("MPE") else "."

    # 1. Cargar datos extraídos
    X_train, X_test, y_train, y_test = ut.get_mpe_features(window_size=1200)

    # 2. Cargar los pesos guardados (.npz) del Mejor Modelo (Penalizado)
    try:
        modelo = np.load(os.path.join(out_dir, 'mejor_modelo.npz'))
        weights_pen = modelo['weights']
        bias_pen = modelo['bias']
    except FileNotFoundError:
        print("Error: No se encontró mejor_modelo.npz. Ejecuta train.py primero.")
        exit()

    # 3. Predicciones
    y_pred_train = predict(X_train, weights_pen, bias_pen)
    y_pred_test = predict(X_test, weights_pen, bias_pen)

    # 4. Calcular métricas y exportar CSVs desglosados (Lógica del compañero)
    f1_train = f1_score(y_train, y_pred_train)
    f1_test = f1_score(y_test, y_pred_test)
    
    pd.DataFrame(confusion_matrix(y_train, y_pred_train)).to_csv(os.path.join(out_dir, 'matriz_confusion_train.csv'), index=False, header=False)
    pd.DataFrame(confusion_matrix(y_test, y_pred_test)).to_csv(os.path.join(out_dir, 'matriz_confusion_test.csv'), index=False, header=False)
    pd.DataFrame({'F1_Train': [f1_train]}).to_csv(os.path.join(out_dir, 'fscores_train.csv'), index=False)
    pd.DataFrame({'F1_Test': [f1_test]}).to_csv(os.path.join(out_dir, 'fscores_test.csv'), index=False)

    print(f"F1-Score Train: {f1_train:.4f} | F1-Score Test: {f1_test:.4f}")

    # 5. Llamar a plot.py para generar el PDF final
    plot.generar_pdf(y_train, y_pred_train, y_test, y_pred_test, f1_train, f1_test, out_dir)
    print(f"¡Evaluación completada! PDF generado exitosamente en la carpeta {out_dir}/")