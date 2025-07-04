# scripts/analisis_ranking_combinaciones.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import logging
from itertools import combinations
from kneed import KneeLocator
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# --- Setup del Path y Logger ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from src import data_processing, config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def analizar_y_rankear_combinaciones():
    """
    Evalúa combinaciones de variables de CALIDAD DE AGUA, calcula un score
    basado en cantidad de datos y correlación, y devuelve un ranking.
    """
    logger.info("Cargando datos base...")
    df_base = data_processing.crear_dataframe_base()
    
    # --- PARÁMETROS DEL ANÁLISIS ---
    RANGO_K_A_PROBAR = range(5, 12) # Probar de 5 a 11 variables de WQ
    W_DATOS = 0.5  # Peso para la cantidad de datos (50%)
    W_CORRELACION = 0.5 # Peso para la correlación (50%)
    # ---------------------------------
    
    # --- CAMBIO CLAVE: Usar solo WATER_QUALITY_FEATURES ---
    logger.info("Analizando combinaciones únicamente de WATER_QUALITY_FEATURES.")
    variables_a_probar = config.WATER_QUALITY_FEATURES
    variables_disponibles = [v for v in variables_a_probar if v in df_base.columns]
    
    # Calcular correlación de todas las variables WQ con el target
    log_target_col = f'{config.TARGET_COLUMN}_log'
    df_base[log_target_col] = np.log1p(df_base[config.TARGET_COLUMN])
    correlaciones = df_base[variables_disponibles + [log_target_col]].corr()[log_target_col].abs().drop(log_target_col)

    resultados_globales = []
    
    for k in RANGO_K_A_PROBAR:
        logger.info(f"Analizando combinaciones para k={k}...")
        
        # Las otras features se añadirán después del filtrado, por lo que no se incluyen aquí
        columnas_completas_fijas = config.RAIN_FEATURES + config.TEMP_FEATURES + config.SOLAR_FEATURES + config.SEASONAL_FEATURES + config.BASIN_FEATURES
        
        combis = combinations(variables_disponibles, k)
        
        for combo in tqdm(list(combis), desc=f"Probando k={k}"):
            # El filtro de filas se basa en el target y la combinación de WQ
            columnas_a_evaluar = [config.TARGET_COLUMN] + list(combo)
            df_filtrado = df_base.dropna(subset=columnas_a_evaluar)
            
            n_filas = len(df_filtrado)
            # El score de correlación solo considera las variables de WQ
            avg_corr = correlaciones[list(combo)].mean()
            
            resultados_globales.append({
                'num_variables_wq': k,
                'combinacion_wq': ', '.join(combo),
                'filas_restantes': n_filas,
                'correlacion_promedio_wq': avg_corr
            })

    if not resultados_globales:
        logger.error("No se pudo generar ningún resultado.")
        return

    df_scores = pd.DataFrame(resultados_globales)
    
    scaler = MinMaxScaler()
    df_scores[['filas_norm', 'corr_norm']] = scaler.fit_transform(df_scores[['filas_restantes', 'correlacion_promedio_wq']])
    df_scores['score_final'] = (W_DATOS * df_scores['filas_norm']) + (W_CORRELACION * df_scores['corr_norm'])
    
    df_scores = df_scores.sort_values(by='score_final', ascending=False).reset_index(drop=True)
    
    output_dir = os.path.join(config.PATHS['OUTPUTS'], 'analisis_combinaciones')
    os.makedirs(output_dir, exist_ok=True)
    ruta_ranking = os.path.join(output_dir, 'ranking_final_combinaciones_wq.csv')
    df_scores.to_csv(ruta_ranking, index=False)
    
    logger.info(f"Ranking final de combinaciones guardado en: {ruta_ranking}")
    print("\n--- MEJORES 15 COMBINACIONES ENCONTRADAS (Rankeadas por Score) ---")
    print(df_scores.head(15).to_string())

if __name__ == "__main__":
    analizar_y_rankear_combinaciones()