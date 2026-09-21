import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import plot # Conecta con el archivo plot.py que armaste en el paso anterior
import utility as ut

def sigmoid(z):
    """Función de activación Sigmoide."""
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

def predict(X, weights, bias, threshold=0.5):
    """Calcula la probabilidad y la convierte en una etiqueta binaria (0 o 1)."""
    z = np.dot(X, weights) + bias
    return (sigmoid(z) >= threshold).astype(int)

if __name__ == "__main__":
    print("--- Iniciando Evaluación del Modelo (MPE) ---")
    
    # 1. Cargar datos de Entrenamiento y Prueba desde utility.py
    # Descomentar cuando utility.py esté programado:
    # X_train, y_train, X_test, y_test = ut.get_mpe_features()
    
    # MOCK DATA: Simulados para validar la ejecución del código
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(40, 5)
    y_test = np.random.randint(0, 2, 40)
    # -----------------------------------------------------------
    
    # 2. Leer los pesos entrenados del CSV generado por train.py
    try:
        df_coef = pd.read_csv('mpe_coeficientes_regresion.csv')
        
        # Seleccionamos el modelo Penalizado como el "Mejor Modelo"
        bias_pen = df_coef['Valor_Penalizado'].iloc[0]
        weights_pen = df_coef['Valor_Penalizado'].iloc[1:].values
        
    except FileNotFoundError:
        print("Error: Ejecuta python train.py primero para generar los pesos.")
        exit()

    # 3. Clasificar los datos
    y_pred_train = predict(X_train, weights_pen, bias_pen)
    y_pred_test = predict(X_test, weights_pen, bias_pen)

    # 4. Calcular F-scores y exportar la Matriz de Confusión numérica (CSV)
    f1_train = f1_score(y_train, y_pred_train)
    f1_test = f1_score(y_test, y_pred_test)
    cm_test = confusion_matrix(y_test, y_pred_test)
    
    pd.DataFrame(cm_test).to_csv('mpe_matriz_confusion_test.csv', index=False, header=False)
    
    print(f"F1-Score (Train): {f1_train:.4f}")
    print(f"F1-Score (Test): {f1_test:.4f}")
    print("Métricas exportadas correctamente.")

    # 5. Llamar a plot.py para generar los PNGs requeridos para tu PDF
    plot.plot_confusion_matrix(y_test, y_pred_test, 'Matriz de Confusión (Test Penalizado)', 'mpe_matriz_confusion.png')
    plot.plot_f_scores(f1_train, f1_test, 'mpe_f_scores.png')
    
    print("Gráficas renderizadas exitosamente.")