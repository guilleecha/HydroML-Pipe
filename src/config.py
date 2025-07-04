# src/config.py
# Unica fuente de verdad para toda la configuracion del proyecto.

import os

# =============================================================================
# 1. RUTAS DEL PROYECTO
# =============================================================================
try:
    src_dir = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(src_dir)
except NameError:
    PROJECT_ROOT = os.path.abspath(os.getcwd())

PATHS = {
    'INPUT_DATA_WQ': os.path.join(PROJECT_ROOT, 'data', 'raw', 'data_wq.xlsx'),
    'INPUT_DATA_PLUV': os.path.join(PROJECT_ROOT, 'data', 'raw', 'pluv_INUMET.txt'),
    'INPUT_DATA_TEMP': os.path.join(PROJECT_ROOT, 'data', 'raw', 'Temp_INUMET.txt'),
    'INPUT_GIS_SHAPEFILES': os.path.join(PROJECT_ROOT, 'data', 'gis', 'shapefiles'),
    'INPUT_DATA_CUENCAS': os.path.join(PROJECT_ROOT, 'data', 'processed', 'cuencas_info_wide.csv'),
    'OUTPUTS': os.path.join(PROJECT_ROOT, 'outputs'),
}

# =============================================================================
# 2. PARAMETROS GENERALES DEL PROYECTO
# =============================================================================
TARGET_COLUMN = 'coliformes_fecales'
DATE_COLUMN = 'date'
METADATA_COLUMNS = ['station_code', 'basin', 'date']

# --- Definicion de Conjuntos de Features ---
WATER_QUALITY_FEATURES = [
    'temp_cent', 'ph_field', 'conductivity', 'dissolve_oxygen_field', 'dbo', 'dqo', 
    'amonio_field', 'total_p', 'total_n', 'ssv', 'sst', 'total_s', 'per_oxida', 
    'tensoactivo', 'nh3', 'plomo', 'cromo', 'isca',
]
RAIN_FEATURES = [
    'precipitacion_promedio_mm', 'prec_acum_3d', 'prec_acum_7d', 'prec_media_7d',
]
BASIN_FEATURES = [
    'basin_area_a', 'mean_slope_of_the_basin_degrees', 'drainage_density_dd',
    'time_of_concentration_-_kirpich_tc', 'main_channel_sinuosity', 'ruggedness_number_rn',
]


SEASONAL_FEATURES = ['seasonal_sin', 'seasonal_cos']

SOLAR_FEATURES = ['horas_sol'] # <-- AÑADIR


TEMP_FEATURES = [
    'temp_ambiente_promedio', 
    'temp_media_3d', 
    'temp_media_7d', 
    'temp_media_14d'
]


ALL_PREDICTOR_VARIABLES = (
    WATER_QUALITY_FEATURES + 
    RAIN_FEATURES + 
    BASIN_FEATURES + 
    SEASONAL_FEATURES + 
    TEMP_FEATURES +
    SOLAR_FEATURES # <-- AÑADIR
)

# --- Parametros de Herencia de Cuencas ---
CUENCAS_HEREDADAS = {
    'AMO0': 'MO1', 'AMO1': 'MO1', 'AMO2': 'MO1',
    'LR1':  'MO1', 'LR2':  'MO1', 'LR3':  'MO1',
}



# =============================================================================
# 4. BATERÍA DE EXPERIMENTOS
# =============================================================================

# Definimos las combinaciones de WQ para mayor claridad
wq_vars_k5 = ['temp_cent', 'ph_field', 'conductivity', 'dissolve_oxygen_field', 'amonio_field']
wq_vars_k8 = ['temp_cent', 'ph_field', 'conductivity', 'dqo', 'amonio_field', 'total_p', 'per_oxida', 'nh3']
wq_vars_k11 = ['temp_cent', 'ph_field', 'conductivity', 'dissolve_oxygen_field', 'dbo', 'dqo', 'amonio_field', 'total_p', 'sst', 'per_oxida', 'nh3']

EXPERIMENTS_TO_RUN = [
    
    {
    'id': 'OPT1_RF_GridSearch_k5',
    'model_name': 'Random Forest',
    'optimization_strategy': 'grid_search',
    'param_grid': {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_leaf': [1, 3, 5],
        'min_samples_split': [2, 5]
    },
    'feature_set': wq_vars_k5 + RAIN_FEATURES + BASIN_FEATURES + SEASONAL_FEATURES + TEMP_FEATURES, # Usar el set óptimo k=8
    'filter_params': {'min_station_samples': 20},
    'validation_strategy': 'time_series_cv', 'time_series_splits': 5,
    'outlier_removal_params': {'apply': False},
    'clustering_params': {'apply': False},
    'generate_eda_plots': True # Generamos plots para este caso
},
    
    {
    'id': 'OPT1_RF_GridSearch_k8',
    'model_name': 'Random Forest',
    'optimization_strategy': 'grid_search',
    'param_grid': {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_leaf': [1, 3, 5],
        'min_samples_split': [2, 5]
    },
    'feature_set': wq_vars_k8 + RAIN_FEATURES + BASIN_FEATURES + SEASONAL_FEATURES + TEMP_FEATURES,
    'filter_params': {'min_station_samples': 20},
    'validation_strategy': 'time_series_cv', 'time_series_splits': 5,
    'outlier_removal_params': {'apply': False},
    'clustering_params': {'apply': False},
    'generate_eda_plots': True # Generamos plots para este caso
},
    
    {
    'id': 'OPT1_RF_GridSearch_k11',
    'model_name': 'Random Forest',
    'optimization_strategy': 'grid_search',
    'param_grid': {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_leaf': [1, 3, 5],
        'min_samples_split': [2, 5]
    },
    'feature_set': wq_vars_k11 + RAIN_FEATURES + BASIN_FEATURES + SEASONAL_FEATURES + TEMP_FEATURES,
    'filter_params': {'min_station_samples': 20},
    'validation_strategy': 'time_series_cv', 'time_series_splits': 5,
    'outlier_removal_params': {'apply': False},
    'clustering_params': {'apply': False},
    'generate_eda_plots': True # Generamos plots para este caso
},
    
    # --- GRUPO P: ANÁLISIS DE SENSIBILIDAD DEL NÚMERO DE FOLDS (k) ---
# Objetivo: Evaluar cómo la elección de k afecta la estabilidad y el
# rendimiento promedio del modelo.

    {
        'id': 'P1_RF_k3_folds',
        'model_name': 'Random Forest',
        'hyperparameters': {'n_estimators': 300, 'random_state': 42},
        'feature_set': wq_vars_k11 + RAIN_FEATURES + BASIN_FEATURES + SEASONAL_FEATURES + TEMP_FEATURES, # Usar el set óptimo k=8
        'filter_params': {'min_station_samples': 20},
        'validation_strategy': 'time_series_cv', 
        'time_series_splits': 3, # <-- k=3
        'outlier_removal_params': {'apply': False},
        'clustering_params': {'apply': False},
        'generate_eda_plots': False
    },
    {
        'id': 'P2_RF_k5_folds',
        'model_name': 'Random Forest',
        'hyperparameters': {'n_estimators': 300, 'random_state': 42},
        'feature_set': wq_vars_k11 + RAIN_FEATURES + BASIN_FEATURES + SEASONAL_FEATURES + TEMP_FEATURES,
        'filter_params': {'min_station_samples': 20},
        'validation_strategy': 'time_series_cv', 
        'time_series_splits': 5, # <-- k=5 (estándar)
        'outlier_removal_params': {'apply': False},
        'clustering_params': {'apply': False},
        'generate_eda_plots': False
    },
    {
        'id': 'P3_RF_k10_folds',
        'model_name': 'Random Forest',
        'hyperparameters': {'n_estimators': 300, 'random_state': 42},
        'feature_set': wq_vars_k11 + RAIN_FEATURES + BASIN_FEATURES + SEASONAL_FEATURES + TEMP_FEATURES,
        'filter_params': {'min_station_samples': 20},
        'validation_strategy': 'time_series_cv', 
        'time_series_splits': 10, # <-- k=10
        'outlier_removal_params': {'apply': False},
        'clustering_params': {'apply': False},
        'generate_eda_plots': False
    },
    
    
    # --- GRUPO DE OPTIMIZACIÓN PARA REDES NEURONALES ---
{
    'id': 'L2_MLP_GridSearch_k8',
    'model_name': 'MLP Neural Network',
    'optimization_strategy': 'grid_search',
    'param_grid': {
        'hidden_layer_sizes': [(32, 16), (64, 32, 16), (50, 50)], # Probamos 3 arquitecturas
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01]  # Probamos 3 niveles de regularización
    },
    # Usamos el mismo feature_set que el experimento campeón
    'feature_set': ['temp_cent', 'ph_field', 'conductivity', 'dqo', 'amonio_field', 'total_p', 'per_oxida', 'nh3'],
    'filter_params': {'min_station_samples': 20},
    'validation_strategy': 'time_series_cv', 'time_series_splits': 5,
    'outlier_removal_params': {'apply': False},
    'clustering_params': {'apply': False},
    'generate_eda_plots': False
},

# --- GRUPO DE OPTIMIZACIÓN PARA MODELO LINEAL REGULARIZADO ---
{
    'id': 'L3_Ridge_GridSearch_k8',
    'model_name': 'Ridge', # <-- Usamos el nuevo modelo
    'optimization_strategy': 'grid_search',
    'param_grid': {
        # Probamos un rango de valores para alpha en escala logarítmica
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0] 
    },
    # Usamos el mismo feature_set que el experimento campeón
    'feature_set': ['temp_cent', 'ph_field', 'conductivity', 'dqo', 'amonio_field', 'total_p', 'per_oxida', 'nh3'],
    'filter_params': {'min_station_samples': 20},
    'validation_strategy': 'time_series_cv', 'time_series_splits': 5,
    'outlier_removal_params': {'apply': False},
    'clustering_params': {'apply': False},
    'generate_eda_plots': False
}


]



# =============================================================================
# 6. PARAMETROS DE REPORTE FINAL
# =============================================================================
REPORTING_PARAMS = {
    # Poner en True para generar la tabla y el grafico que compara todos los
    # experimentos al final de la ejecucion. Poner en False para omitir este paso.
    'generate_final_comparison_report': True
}