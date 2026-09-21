import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import plot
import utility as ut
import os, json

def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

if __name__ == "__main__":
    out_dir = os.path.dirname(__file__)
    with open(os.path.join(out_dir, 'parametros.json'), 'r') as f: params = json.load(f)
    
    # Cargar matriz de transformación del train
    modelo = np.load(os.path.join(out_dir, 'mejor_modelo.npz'))
    mean, std = modelo['mean'], modelo['std']
    w_p, b_p = modelo['w_p'], modelo['b_p']
    
    X_tr_raw, X_ts_raw, y_tr, y_ts = ut.get_features(window_size=params["window_size"])
    X_tr, X_ts = (X_tr_raw - mean) / std, (X_ts_raw - mean) / std
    
    y_p_tr = (sigmoid(np.dot(X_tr, w_p) + b_p) >= 0.5).astype(int)
    y_p_ts = (sigmoid(np.dot(X_ts, w_p) + b_p) >= 0.5).astype(int)

    # Entregables con encabezados claros
    pd.DataFrame(confusion_matrix(y_tr, y_p_tr), columns=['Pred_Normal','Pred_Fallo'], index=['Real_Normal','Real_Fallo']).to_csv(os.path.join(out_dir, 'matriz_confusion_train.csv'))
    pd.DataFrame(confusion_matrix(y_ts, y_p_ts), columns=['Pred_Normal','Pred_Fallo'], index=['Real_Normal','Real_Fallo']).to_csv(os.path.join(out_dir, 'matriz_confusion_test.csv'))
    
    rep_tr = pd.DataFrame(classification_report(y_tr, y_p_tr, output_dict=True)).T
    rep_ts = pd.DataFrame(classification_report(y_ts, y_p_ts, output_dict=True)).T
    rep_tr.to_csv(os.path.join(out_dir, 'fscores_train.csv'))
    rep_ts.to_csv(os.path.join(out_dir, 'fscores_test.csv'))
    
    plot.generar_pdf(y_tr, y_p_tr, y_ts, y_p_ts, rep_tr.loc['macro avg','f1-score'], rep_ts.loc['macro avg','f1-score'], out_dir)