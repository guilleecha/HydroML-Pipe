# scripts/analizar_clusters.py
# Realiza el análisis de los clusters generados, guardando los resultados
# numéricos y visuales en la carpeta outputs/clusters/.

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Añadimos el directorio raíz del proyecto al path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src import config

# Configurar un logger básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

logger = logging.getLogger(__name__)


def analizar_y_guardar_resultados_cluster(k):
    """
    Carga los datos de clusters y cuencas, calcula perfiles y genera gráficos,
    guardando todos los resultados en una carpeta.
    """
    logger.info(f"--- Iniciando análisis para k={k} clusters ---")

    # Definir la carpeta de salida para este análisis
    output_cluster_dir = os.path.join(config.PATHS['OUTPUTS'], 'clusters')
    
    # --- Paso 1: Cargar y Fusionar los Datos ---
    path_clusters = os.path.join(output_cluster_dir, f'station_clusters_k{k}.csv')
    try:
        df_clusters = pd.read_csv(path_clusters)
    except FileNotFoundError:
        logger.error(f"Error: No se encontró el archivo de clusters en {path_clusters}")
        logger.error("Asegúrate de haber corrido un experimento con clustering (k={k}) primero.")
        return

    df_cuencas_info = pd.read_csv(config.PATHS['INPUT_DATA_CUENCAS'])
    df_analisis = pd.merge(df_clusters, df_cuencas_info, on='station_code')
    logger.info("Datos de clusters y cuencas fusionados correctamente.")

    # --- Paso 2: Análisis Numérico y Guardado de Tabla ---
    perfil_clusters = df_analisis.groupby('cluster_id')[config.BASIN_FEATURES].mean().round(2)
    
    path_tabla_perfil = os.path.join(output_cluster_dir, f'perfil_promedio_clusters_k{k}.csv')
    perfil_clusters.to_csv(path_tabla_perfil)
    
    logger.info(f"Perfil promedio de clusters guardado en: {path_tabla_perfil}")
    print("\n--- Perfil Promedio de Cada Clúster ---")
    print(perfil_clusters)
    
    # --- Paso 3: Análisis Visual y Guardado de Gráficos ---
    logger.info("Generando y guardando gráficos de distribución por cluster...")
    for feature in config.BASIN_FEATURES:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_analisis, x='cluster_id', y=feature, palette='viridis')
        plt.title(f'Distribución de "{feature}" por Clúster (k={k})', fontsize=16)
        plt.xlabel('ID del Clúster', fontsize=12)
        plt.ylabel(f'Valor de {feature}', fontsize=12)
        plt.tight_layout()
        
        # Guardar el gráfico
        filename = f'boxplot_k{k}_{feature}.png'
        path_grafico = os.path.join(output_cluster_dir, filename)
        plt.savefig(path_grafico)
        plt.close() # Cerramos la figura para no mostrarla en la consola
        logger.info(f"  - Gráfico guardado: {filename}")
        
    logger.info("--- Análisis de clusters finalizado. ---")


if __name__ == "__main__":
    # Define aquí para qué valor de 'k' quieres generar el análisis
    # (debe ser uno de los que corriste en tus experimentos)
    K_PARA_ANALIZAR = 3
    
    analizar_y_guardar_resultados_cluster(k=K_PARA_ANALIZAR)