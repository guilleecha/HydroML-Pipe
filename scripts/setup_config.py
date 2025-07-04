# scripts/setup_config.py

import os
import sys
import pandas as pd
import logging

# --- Setup del Path y Logger ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Importamos las funciones de ayuda que ya tenemos para leer y limpiar
from src.data_processing import _limpiar_nombres_columnas, _procesar_datos_lluvia, _procesar_datos_temperatura

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- Esta es una versión simplificada de la configuración de PATHS para este script ---
PATHS = {
    'INPUT_DATA_WQ': os.path.join(project_root, 'data', 'raw', 'data_wq.xlsx'),
    'INPUT_DATA_PLUV': os.path.join(project_root, 'data', 'raw', 'pluv_INUMET.txt'),
    'INPUT_DATA_TEMP': os.path.join(project_root, 'data', 'raw', 'Temp_INUMET.txt'),
    'INPUT_DATA_CUENCAS': os.path.join(project_root, 'data', 'processed', 'cuencas_info_wide.csv'),
    'CONFIG_FILE_OUTPUT': os.path.join(project_root, 'src', 'config.py')
}

def generar_configuracion():
    """
    Lee los archivos de datos crudos, detecta las columnas y genera un
    archivo config.py completo y funcional.
    """
    logger.info("Iniciando la generación automática del archivo de configuración...")

    # --- 1. Detectar Variables de Calidad de Agua ---
    df_wq = pd.read_excel(PATHS['INPUT_DATA_WQ'], sheet_name='data', engine='openpyxl')
    df_wq = df_wq.pipe(_limpiar_nombres_columnas)
    
    # Excluimos las columnas que sabemos que son metadatos
    metadata_cols = ['station_code', 'basin', 'date', 'coliformes_fecales']
    water_quality_features = [col for col in df_wq.columns if col not in metadata_cols]
    
    # --- 2. Detectar Variables de Cuenca ---
    df_cuencas = pd.read_csv(PATHS['INPUT_DATA_CUENCAS'])
    df_cuencas = df_cuencas.pipe(_limpiar_nombres_columnas)
    basin_features = [col for col in df_cuencas.columns if col not in ['station_code', 'basin', 'cuenca']]
    
    # --- 3. Definir Variables Derivadas (Lluvia, Temp, etc.) ---
    # Estas se definen estáticamente porque se crean en el preprocesamiento
    rain_features = ['precipitacion_promedio_mm', 'prec_acum_3d', 'prec_acum_7d', 'prec_media_7d']
    temp_features = ['temp_ambiente_promedio', 'temp_media_3d', 'temp_media_7d', 'temp_media_14d']
    solar_features = ['horas_sol']
    seasonal_features = ['seasonal_sin', 'seasonal_cos']
    
    # --- 4. Ensamblar el Contenido del Archivo config.py ---
    contenido_config = f"""
# src/config.py
# ARCHIVO AUTO-GENERADO POR setup_config.py
# Unica fuente de verdad para toda la configuracion del proyecto.

import os

# --- 1. RUTAS DEL PROYECTO ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHS = {{
    'INPUT_DATA_WQ': os.path.join(PROJECT_ROOT, 'data', 'raw', 'data_wq.xlsx'),
    'INPUT_DATA_PLUV': os.path.join(PROJECT_ROOT, 'data', 'raw', 'pluv_INUMET.txt'),
    'INPUT_DATA_TEMP': os.path.join(PROJECT_ROOT, 'data', 'raw', 'Temp_INUMET.txt'),
    'INPUT_DATA_CUENCAS': os.path.join(PROJECT_ROOT, 'data', 'processed', 'cuencas_info_wide.csv'),
    'OUTPUTS': os.path.join(PROJECT_ROOT, 'outputs'),
}}

# --- 2. PARAMETROS GENERALES ---
TARGET_COLUMN = 'coliformes_fecales'
DATE_COLUMN = 'date'
METADATA_COLUMNS = ['station_code', 'basin', 'date']
CUENCAS_HEREDADAS = {{
    'AMO0': 'MO1', 'AMO1': 'MO1', 'AMO2': 'MO1',
    'LR1': 'MO1', 'LR2': 'MO1', 'LR3': 'MO1'
}}
REPORTING_PARAMS = {{
    'generate_final_comparison_report': True
}}

# --- 3. LISTAS DE VARIABLES (AUTO-GENERADAS) ---
WATER_QUALITY_FEATURES = {water_quality_features}
BASIN_FEATURES = {basin_features}
RAIN_FEATURES = {rain_features}
TEMP_FEATURES = {temp_features}
SOLAR_FEATURES = {solar_features}
SEASONAL_FEATURES = {seasonal_features}

ALL_PREDICTOR_VARIABLES = (
    WATER_QUALITY_FEATURES + RAIN_FEATURES + BASIN_FEATURES + 
    SEASONAL_FEATURES + TEMP_FEATURES + SOLAR_FEATURES
)

# --- 4. BATERÍA DE EXPERIMENTOS ---
# Define aquí tus experimentos. Se incluye un ejemplo para empezar.
EXPERIMENTS_TO_RUN = [
    {{
        'id': 'Ejemplo_RF_k8',
        'model_name': 'Random Forest',
        'hyperparameters': {{'n_estimators': 200, 'random_state': 42}},
        'feature_set': ['temp_cent', 'ph_field', 'conductivity', 'dqo', 'amonio_field', 'total_p', 'per_oxida', 'nh3'] + RAIN_FEATURES + BASIN_FEATURES + SEASONAL_FEATURES + TEMP_FEATURES + SOLAR_FEATURES,
        'filter_params': {{'min_station_samples': 20}},
        'validation_strategy': 'time_series_cv', 'time_series_splits': 5,
        'outlier_removal_params': {{'apply': False}},
        'clustering_params': {{'apply': False}},
        'generate_eda_plots': True
    }},
]
"""
    
    # --- 5. Escribir el Archivo ---
    try:
        with open(PATHS['CONFIG_FILE_OUTPUT'], 'w', encoding='utf-8') as f:
            f.write(contenido_config.strip())
        logger.info(f"¡Éxito! El archivo 'config.py' ha sido generado en: {PATHS['CONFIG_FILE_OUTPUT']}")
    except Exception as e:
        logger.error(f"No se pudo escribir el archivo de configuración. Error: {e}")

if __name__ == "__main__":
    generar_configuracion()