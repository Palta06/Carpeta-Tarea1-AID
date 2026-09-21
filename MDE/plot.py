import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

def plot_convergence(csv_filename):
    """Grafica la curva de convergencia comparando mGD normal vs penalizado para MDE."""
    try:
        df = pd.read_csv(csv_filename)
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['Epoch'], df['Loss_Normal'], label='Entropía Cruzada Normal', linewidth=2)
        plt.plot(df['Epoch'], df['Loss_Penalizada'], label='Entropía Cruzada Penalizada', linewidth=2, linestyle='--')
        
        plt.title('Curva de Convergencia MDE: Descenso de Gradiente con Momentum (mGD)')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida (Loss)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('mde_curva_convergencia.png', dpi=300)
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {csv_filename}. Asegúrate de ejecutar train.py primero.")

def plot_confusion_matrix(y_true, y_pred, title, filename):
    """Genera y guarda un mapa de calor para la matriz de confusión de MDE."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False,
                xticklabels=['Normal (0)', 'Fallo (1)'],
                yticklabels=['Normal (0)', 'Fallo (1)'])
    plt.title(title)
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción del Modelo')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

def plot_f_scores(f1_train, f1_test, filename):
    """Grafica una comparativa en barras de los F-scores de Train y Test para MDE."""
    labels = ['Entrenamiento (Train)', 'Prueba (Test)']
    scores = [f1_train, f1_test]
    
    plt.figure(figsize=(7, 5))
    # Cambiamos los colores para diferenciar visualmente MDE de MPE en tu PDF
    bars = plt.bar(labels, scores, color=['#E6842A', '#137B80'], width=0.5)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 4), ha='center', va='bottom', fontweight='bold')
    
    plt.ylim(0, 1.1) 
    plt.title('Rendimiento del Modelo MDE: F-score')
    plt.ylabel('F-score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

if __name__ == "__main__":
    print("--- Generando gráficas para el PDF (MDE) ---")
    
    # 1. Graficar Convergencia
    plot_convergence('mde_convergencia_mGD.csv')
    
    # MOCK DATA temporal (recuerda borrar esto cuando conectemos con tst.py)
    y_true_mock = np.random.randint(0, 2, 100)
    y_pred_mock = y_true_mock.copy()
    y_pred_mock[::8] = 1 - y_pred_mock[::8] 
    
    f1_train_mock = f1_score(y_true_mock, y_pred_mock)
    f1_test_mock = f1_train_mock - 0.04 
    # -----------------------------------------------------------
    
    # 2. Graficar Matrices de Confusión
    plot_confusion_matrix(y_true_mock, y_pred_mock, 'Matriz de Confusión MDE (Datos Simulados)', 'mde_matriz_confusion.png')
    
    # 3. Graficar F-scores
    plot_f_scores(f1_train_mock, f1_test_mock, 'mde_f_scores.png')
    
    print("\nGráficas MDE guardadas como imágenes .png en tu carpeta.")