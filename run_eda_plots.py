# run_eda_plots.py
# Script para generar el conjunto de graficos de Analisis Exploratorio de Datos (EDA).

import pandas as pd
import os
import logging
import sys

# Importamos desde nuestra nueva estructura 'src'
from src import config, utils, logger_config, analysis_tools

# Configurar un logger basico para ver los mensajes en la consola.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger('PipelineLogger')

def main():
    """
    Funcion principal que orquesta la generacion de graficos EDA.
    """
    logger.info("--- INICIANDO GENERACION DE GRAFICOS DE ANALISIS EXPLORATORIO (EDA) ---")
    
    # 1. Asegurar que los directorios de salida existan
    utils.setup_directorios()
    plot_output_path = os.path.join(config.PATHS['OUTPUTS'], 'visualizations', 'eda')
    os.makedirs(plot_output_path, exist_ok=True)
    
    # 2. Cargar los datos. Usaremos el "DataFrame Base" que genera el pipeline de datos.
    # Esto asegura que analizamos los datos ya limpios y filtrados.
    try:
        df_base = data_processing.crear_dataframe_base()
        logger.info("DataFrame Base cargado para el analisis.")
    except Exception as e:
        logger.error(f"No se pudo crear el DataFrame Base. Error: {e}")
        logger.error("Ejecuta primero 'scripts/generar_info_cuencas.py' si es necesario.")
        return

    # 3. Lista de variables para las que queremos generar los graficos
    variables_series_temporales = [config.TARGET_COLUMN, 'dbo', 'conductivity']
    logger.info(f"Generando {len(variables_series_temporales)} graficos de series temporales con outliers...")
    for var in variables_series_temporales:
        analysis_tools.plot_serie_temporal_con_outliers(df_base, var, plot_output_path)

    # 4. Generar el mapa de calor de correlacion
    # Usaremos solo las features de calidad de agua y lluvia para este grafico
    features_para_corr = config.WATER_QUALITY_FEATURES + config.RAIN_FEATURES
    analysis_tools.plot_correlation_heatmap(df_base, features_para_corr, plot_output_path)

    logger.info(f"\n✅ Graficos EDA guardados en la carpeta: '{plot_output_path}'")


if __name__ == "__main__":
    main()