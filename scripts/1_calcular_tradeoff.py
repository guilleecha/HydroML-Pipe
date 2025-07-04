# scripts/1_calcular_tradeoff.py

import pandas as pd
import sys
import os
import logging
from itertools import combinations
from tqdm import tqdm

# --- Setup ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from src import data_processing, config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def encontrar_top_combinaciones(df, variables_disponibles, k, top_n=10):
    """
    Encuentra el ranking de las 'top_n' mejores combinaciones de 'k' variables
    que maximizan la cantidad de datos restantes.
    """
    logger.info(f"Buscando las mejores {top_n} combinaciones para k={k}...")
    resultados_k = []
    
    combis = combinations(variables_disponibles, k)
    
    for combo in tqdm(list(combis), desc=f"Probando k={k}"):
        columnas_a_evaluar = [config.TARGET_COLUMN] + list(combo)
        df_filtrado = df.dropna(subset=columnas_a_evaluar)
        n_filas = len(df_filtrado)
        resultados_k.append({'combinacion': combo, 'filas_restantes': n_filas})

    # Ordenar todas las combinaciones por filas restantes y quedarse con el top N
    df_ranking = pd.DataFrame(resultados_k)
    df_ranking = df_ranking.sort_values(by='filas_restantes', ascending=False).head(top_n)
    
    return df_ranking

def ejecutar_analisis_de_combinaciones():
    logger.info("Cargando datos base...")
    df_base = data_processing.crear_dataframe_base()
    variables_disponibles = [v for v in config.WATER_QUALITY_FEATURES if v in df_base.columns]
    
    output_dir = os.path.join(config.PATHS['OUTPUTS'], 'analisis_combinaciones')
    os.makedirs(output_dir, exist_ok=True)
    
    RANGO_K_A_PROBAR = range(2, 18)
    TOP_N_RANKING = 10

    resumen_mejores_opciones = []
    
    for k in RANGO_K_A_PROBAR:
        df_ranking_k = encontrar_top_combinaciones(df_base, variables_disponibles, k, top_n=TOP_N_RANKING)
        
        # --- MEJORA: Hacemos el post-procesamiento aquí ---
        # Guardamos la mejor opción para el gráfico de trade-off
        mejor_opcion_k = df_ranking_k.iloc[0]
        
        # Contamos las estaciones para la mejor combinación
        columnas_mejor_combo = [config.TARGET_COLUMN] + list(mejor_opcion_k['combinacion'])
        df_filtrado_mejor = df_base.dropna(subset=columnas_mejor_combo)
        
        resumen_mejores_opciones.append({
            'num_variables': k,
            'mejor_combinacion': ', '.join(mejor_opcion_k['combinacion']),
            'filas_maximas': mejor_opcion_k['filas_restantes'],
            'estaciones_restantes': df_filtrado_mejor['station_code'].nunique()
        })

        # Convertimos la columna de tuplas a string ANTES de guardar el CSV
        df_ranking_k['combinacion'] = df_ranking_k['combinacion'].apply(lambda x: ', '.join(x))
        ruta_ranking_k = os.path.join(output_dir, f'top_{TOP_N_RANKING}_combinaciones_k{k}.csv')
        df_ranking_k.to_csv(ruta_ranking_k, index=False)
        logger.info(f"Ranking para k={k} guardado en: {ruta_ranking_k}")

    # Crear y guardar el DataFrame resumen final
    df_tradeoff = pd.DataFrame(resumen_mejores_opciones)
    ruta_tradeoff = os.path.join(output_dir, 'analisis_tradeoff_optimo.csv')
    df_tradeoff.to_csv(ruta_tradeoff, index=False)
    logger.info(f"\nTabla de trade-off con las mejores opciones guardada en: {ruta_tradeoff}")
    print(df_tradeoff.to_string())
    logger.info("--- CÁLCULO DE COMBINACIONES FINALIZADO ---")

if __name__ == "__main__":
    ejecutar_analisis_de_combinaciones()