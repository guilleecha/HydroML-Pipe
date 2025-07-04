# scripts/run_cross_validation.py

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src import config
from src.train import preparar_datos_modelo, impute_and_scale
import logging

logger = logging.getLogger('PipelineLogger')

def ejecutar_validacion_cruzada():
    logger.info("--- INICIANDO VALIDACIÓN CRUZADA (K-FOLD) ---")
    
    # 1. Cargar y preparar los datos como siempre
    ruta_datos = f"{config.PATHS['OUTPUTS_DATA']}/datos_para_modelo.csv"
    df = pd.read_csv(ruta_datos)
    X, y = preparar_datos_modelo(df)

    # 2. Definir los modelos a probar
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=config.ML_PARAMS['random_state'])
    }
    
    # 3. Definir la estrategia de Cross-Validation
    cv = KFold(n_splits=5, shuffle=True, random_state=config.ML_PARAMS['random_state'])

    # 4. Iterar y evaluar cada modelo
    for name, model in models.items():
        logger.info(f"--- Evaluando modelo: {name} ---")
        
        # cross_val_score entrena y evalúa el modelo 5 veces automáticamente
        # Usamos 'neg_root_mean_squared_error' para que el error sea comparable
        # Es negativo porque la convención es maximizar, así que se maximiza un valor menos negativo.
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
        
        # Convertimos los scores a positivo y calculamos la media y desviación estándar
        rmse_scores = -scores
        logger.info(f"Scores de RMSE en cada Fold: {np.round(rmse_scores, 2)}")
        logger.info(f"RMSE Promedio: {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})")

if __name__ == '__main__':
    ejecutar_validacion_cruzada()