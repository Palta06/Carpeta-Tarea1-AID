import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages
import os
import json

def generar_pdf(y_train, y_pred_train, y_test, y_pred_test, f1_train, f1_test, out_dir="."):
    pdf_path = os.path.join(out_dir, 'graficas_mejor_modelo_MPE.pdf')
    csv_conv = os.path.join(out_dir, 'mpe_convergencia_mGD.csv')
    json_path = os.path.join(out_dir, 'parametros.json')

    with PdfPages(pdf_path) as pdf:
        # Página 1: Lista de parámetros (Texto)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                params = json.load(f)
            
            fig0 = plt.figure(figsize=(8, 6))
            plt.axis('off') # Apagamos los ejes para que parezca una hoja en blanco
            
            texto_params = "Lista de parámetros del Mejor Modelo:\n" + "-"*50 + "\n\n"
            for key, value in params.items():
                texto_params += f"• {key}: {value}\n\n"
                
            plt.text(0.05, 0.95, texto_params, fontsize=11, family='monospace', va='top', ha='left', wrap=True)
            pdf.savefig(fig0)
            plt.close(fig0)
        except Exception as e:
            print(f"Advertencia: No se pudo cargar el JSON para el PDF: {e}")

        # Página 2: Curva de Convergencia
        try:
            df = pd.read_csv(csv_conv)
            fig1 = plt.figure(figsize=(8, 6))
            plt.plot(df['Epoch'], df['Loss_Normal'], label='Entropía Cruzada Normal')
            plt.plot(df['Epoch'], df['Loss_Penalizada'], label='Entropía Cruzada Penalizada', linestyle='--')
            plt.title('Curva de Convergencia MPE')
            plt.xlabel('Épocas')
            plt.ylabel('Pérdida')
            plt.legend()
            plt.grid(True, alpha=0.3)
            fig1.savefig(os.path.join(out_dir, 'pagina1.png'))
            pdf.savefig(fig1)
            plt.close(fig1)
        except Exception as e:
            print(f"No se pudo graficar convergencia: {e}")

        # Página 3: Matriz Train
        fig2 = plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_train, y_pred_train), annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Matriz de Confusión (Entrenamiento)')
        fig2.savefig(os.path.join(out_dir, 'pagina2.png'))
        pdf.savefig(fig2)
        plt.close(fig2)

        # Página 4: Matriz Test
        fig3 = plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Matriz de Confusión (Prueba)')
        fig3.savefig(os.path.join(out_dir, 'pagina3.png'))
        pdf.savefig(fig3)
        plt.close(fig3)

        # Página 5: F-scores
        fig4 = plt.figure(figsize=(6, 5))
        bars = plt.bar(['Train', 'Test'], [f1_train, f1_test], color=['#4C72B0', '#55A868'], width=0.5)
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.4f}', ha='center', fontweight='bold')
        plt.ylim(0, 1.1)
        plt.title('F1-Scores MPE')
        fig4.savefig(os.path.join(out_dir, 'pagina4.png'))
        pdf.savefig(fig4)
        plt.close(fig4)