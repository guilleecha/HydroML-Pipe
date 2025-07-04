# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 17:21:55 2025

@author: USUARIO
"""

# Parámetros de análisis
ANALYSIS_PARAMS = {
    'variables_principales': [
        'temp_cent', 'ph_field', 'conductivity', 'dissolve_oxygen_field',
        'dbo', 'dqo' , 'amonio_field', 'total_p','total_n', 'ssv','sst', 'total_s', 
        'per_oxida','tensoactivo','nh3','coliformes_fecales','plomo','cromo',
        'isca'
        
    ],
    'umbral_faltantes': 0.99,
    'crs_proyecto': 'EPSG:32721'
}


# Configuración de visualización
PLOT_SETTINGS = {
    'style': 'ggplot',
    'figsize': (12, 6),
    'paleta_cuencas': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
}


MODEL_SETTINGS = {
    'baseline': 'LinearRegression',
    'advanced': 'RandomForestRegressor',
    'test_size': 0.2,
    'random_state': 42
}

# =============================================================================
# PARÁMETROS DE MACHINE LEARNING
# =============================================================================

ML_PARAMS = {
    'TARGET': 'coliformes_fecales',
    'FEATURES': [
        'temp_cent', 'ph_field', 'conductivity', 'dissolve_oxygen_field', 'sst', 'dbo',
        'Basin Area (A)', 'Mean slope of the Basin (degrees)', 'Drainage Density (Dd)'
    ],
    'TEST_SIZE': 0.2, # Proporción de datos para el conjunto de prueba
    'RANDOM_STATE': 42, # Para reproducibilidad
    'RF_N_ESTIMATORS': 100 # Número de árboles en el Random Forest
}