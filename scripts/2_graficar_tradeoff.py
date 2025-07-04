# scripts/2_graficar_tradeoff.py

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import logging
import tikzplotlib

# --- Setup ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from src import config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def graficar_curva_tradeoff():
    """
    Lee el archivo de resultados del análisis de trade-off y genera el gráfico final.
    """
    output_dir = os.path.join(config.PATHS['OUTPUTS'], 'analisis_combinaciones')
    ruta_datos = os.path.join(output_dir, 'analisis_tradeoff_optimo.csv')

    try:
        df_tradeoff = pd.read_csv(ruta_datos)
    except FileNotFoundError:
        logger.error(f"No se encontró el archivo de datos en {ruta_datos}.")
        logger.error("Por favor, ejecuta primero '1_calcular_tradeoff.py'.")
        return

    logger.info("Generando gráfico de sensibilidad con estilo B&W...")
    plt.style.use('grayscale')
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Eje izquierdo para las filas (línea sólida con marcadores redondos)
    p1, = ax1.plot(df_tradeoff['num_variables'], df_tradeoff['filas_maximas'], 
                   marker='o', linestyle='-', color='black', label='Máximo de Filas Posible')
    ax1.set_xlabel('Número de Variables de Calidad de Agua en la Combinación', fontsize=14)
    ax1.set_ylabel('Número de Filas de Datos Restantes', fontsize=14)
    
    # Eje derecho para las estaciones (línea punteada con marcadores cuadrados)
    ax2 = ax1.twinx()
    p2, = ax2.plot(df_tradeoff['num_variables'], df_tradeoff['estaciones_restantes'], 
                   marker='s', linestyle=':', color='black', label='Estaciones Restantes')
    ax2.set_ylabel('Número de Estaciones Restantes', fontsize=14)

    # Combinamos las leyendas en una sola caja
    ax1.legend(handles=[p1, p2], loc='upper right')
    
    plt.title('Curva de Trade-off Óptima: Datos vs. Variables', fontsize=16)
    fig.tight_layout()
    
    # Guardado en ambos formatos
    ruta_png = os.path.join(output_dir, 'grafico_tradeoff_optimo.png')
    ruta_tex = os.path.join(output_dir, 'grafico_tradeoff_optimo.tex')
    plt.savefig(ruta_png)
    tikzplotlib.save(ruta_tex)
    logger.info(f"Gráfico de trade-off guardado en: {ruta_png} y .tex")
    
    plt.show()

if __name__ == "__main__":
    graficar_curva_tradeoff()