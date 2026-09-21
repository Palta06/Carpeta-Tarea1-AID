import subprocess
import sys

def ejecutar_script(ruta):
    """Ejecuta un script de Python y muestra su salida en la terminal."""
    print(f"\n{'='*60}\n>>> Ejecutando: {ruta}\n{'='*60}")
    # Ejecuta el archivo usando el mismo intérprete de Python actual
    proceso = subprocess.run([sys.executable, ruta])
    if proceso.returncode != 0:
        print(f"\n[ERROR] Falló la ejecución de {ruta}. Deteniendo el pipeline.")
        sys.exit(1)

if __name__ == "__main__":
    print("INICIANDO GENERACIÓN DE ENTREGABLES - TAREA #1")
    
    # 1. Flujo de Entropía Multi-escala de Permutación (MPE)
    ejecutar_script("MPE/train.py") # Entrena el modelo y guarda los CSV de pesos
    ejecutar_script("MPE/tst.py")   # Evalúa el modelo y guarda las gráficas PNG
    
    # 2. Flujo de Entropía Multi-escala de Dispersión (MDE)
    ejecutar_script("MDE/train.py")
    ejecutar_script("MDE/tst.py")
    
    print("\n" + "="*60)
    print("¡PIPELINE COMPLETADO CON ÉXITO!")
    print("Tus archivos CSV y PNG están listos en las carpetas MPE y MDE.")
    print("="*60)