import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages
import os

def generar_pdf(y_train, y_pred_train, y_test, y_pred_test, f1_train, f1_test, out_dir="."):
    pdf_path = os.path.join(out_dir, 'graficas_mejor_modelo_MPE.pdf')
    csv_conv = os.path.join(out_dir, 'mpe_convergencia_mGD.csv')

    with PdfPages(pdf_path) as pdf:
        # Página 1: Curva de Convergencia
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

        # Página 2: Matriz Train
        fig2 = plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_train, y_pred_train), annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Matriz de Confusión (Entrenamiento)')
        fig2.savefig(os.path.join(out_dir, 'pagina2.png'))
        pdf.savefig(fig2)
        plt.close(fig2)

        # Página 3: Matriz Test
        fig3 = plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Matriz de Confusión (Prueba)')
        fig3.savefig(os.path.join(out_dir, 'pagina3.png'))
        pdf.savefig(fig3)
        plt.close(fig3)

        # Página 4: F-scores
        fig4 = plt.figure(figsize=(6, 5))
        bars = plt.bar(['Train', 'Test'], [f1_train, f1_test], color=['#4C72B0', '#55A868'], width=0.5)
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.4f}', ha='center', fontweight='bold')
        plt.ylim(0, 1.1)
        plt.title('F1-Scores MPE')
        fig4.savefig(os.path.join(out_dir, 'pagina4.png'))
        pdf.savefig(fig4)
        plt.close(fig4)