import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages
import os, json

def generar_pdf(y_tr, y_p_tr, y_ts, y_p_ts, f1_tr, f1_ts, out_dir):
    with PdfPages(os.path.join(out_dir, 'graficas_mejor_modelo_MPE.pdf')) as pdf:
        # Página 1: Parámetros y descripciones
        with open(os.path.join(out_dir, 'parametros.json'), 'r') as f: params = json.load(f)
        desc = {"metodo": "Algoritmo de Extracción", "window_size": "Tamaño de ventana en muestras", 
                "lambda_penalty": "Fuerza de regularización L2", "learning_rate": "Tasa de aprendizaje", "epochs": "Iteraciones del mGD"}
        
        fig0 = plt.figure(figsize=(8, 6))
        plt.axis('off')
        txt = "Parámetros del Mejor Modelo MPE:\n" + "-"*40 + "\n\n"
        for k, v in params.items(): txt += f"• {k} = {v}\n  ({desc.get(k, '')})\n\n"
        plt.text(0.05, 0.95, txt, fontsize=12, family='monospace', va='top')
        pdf.savefig(fig0); plt.close(fig0)

        # Página 2: Convergencia
        df = pd.read_csv(os.path.join(out_dir, 'mpe_convergencia_mGD.csv'))
        fig1 = plt.figure(figsize=(8, 6))
        plt.plot(df['Epoch'], df['Loss_Normal'], label='Cross-Entropy Normal')
        plt.plot(df['Epoch'], df['Loss_Penalizada'], label='Cross-Entropy Penalizada', linestyle='--')
        plt.title('Curva de Convergencia MPE'); plt.xlabel('Épocas'); plt.ylabel('Pérdida'); plt.legend(); plt.grid(True, alpha=0.3)
        pdf.savefig(fig1); plt.close(fig1)

        # Páginas 3 y 4: Matrices (Train y Test)
        for y_true, y_pred, title in [(y_tr, y_p_tr, 'Entrenamiento'), (y_ts, y_p_ts, 'Prueba')]:
            fig = plt.figure(figsize=(6, 5))
            sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Normal','Fallo'], yticklabels=['Normal','Fallo'])
            plt.title(f'Matriz de Confusión MPE ({title})')
            pdf.savefig(fig); plt.close(fig)

        # Página 5: F-scores Macro
        fig4 = plt.figure(figsize=(6, 5))
        bars = plt.bar(['Train (Macro)', 'Test (Macro)'], [f1_tr, f1_ts], color=['#4C72B0', '#55A868'])
        for bar in bars: plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01, f'{bar.get_height():.4f}', ha='center')
        plt.ylim(0, 1.1); plt.title('F1-Scores (Macro Avg) MPE')
        pdf.savefig(fig4); plt.close(fig4)