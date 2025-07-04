# scripts/run_preliminary_eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import logging
import numpy as np
import tikzplotlib # <-- Importamos la nueva librería

# --- Setup del Path y Logger ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src import data_processing, config, analysis_tools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def plot_data_completeness(df, output_dir):
    """
    Calcula y grafica el porcentaje de datos no nulos, guardando en PNG y TikZ.
    """
    logger.info("Calculando completitud de datos para variables de calidad de agua...")
    
    completeness = df[config.WATER_QUALITY_FEATURES].notna().mean() * 100
    completeness = completeness.sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=completeness.values, y=completeness.index, palette='viridis')
    plt.title('Porcentaje de Datos Disponibles por Variable de Calidad de Agua', fontsize=16)
    plt.xlabel('Completitud de Datos (%)', fontsize=12)
    plt.ylabel('Variable', fontsize=12)
    plt.tight_layout()
    
    # --- Guardado en ambos formatos ---
    ruta_png = os.path.join(output_dir, 'eda_completitud_variables.png')
    ruta_tex = os.path.join(output_dir, 'eda_completitud_variables.tex')
    plt.savefig(ruta_png)
    tikzplotlib.save(ruta_tex)
    logger.info(f"Gráfico de completitud guardado en: {ruta_png} y .tex")
    plt.show()

def plot_target_correlation(df, feature_list, target_col, output_dir):
    """
    Calcula y grafica la correlación con el target, guardando en PNG y TikZ.
    """
    logger.info(f"Generando gráfico de correlación con la variable objetivo '{target_col}'...")
    df_corr = df.copy()
    log_target_col = f'{target_col}_log'
    df_corr[log_target_col] = np.log1p(df_corr[target_col])

    # Nos aseguramos de que solo usamos columnas numéricas
    numeric_features = [f for f in feature_list if pd.api.types.is_numeric_dtype(df_corr[f])]
    correlations = df_corr[numeric_features + [log_target_col]].corr()
    
    corr_con_target = correlations[log_target_col].drop(log_target_col).sort_values(ascending=False)

    plt.figure(figsize=(10, 12)) # Hacemos el gráfico más alto para que entren las etiquetas
    sns.barplot(x=corr_con_target.values, y=corr_con_target.index, palette='coolwarm_r')
    plt.title(f'Correlación de Pearson con {log_target_col}', fontsize=16)
    plt.xlabel('Coeficiente de Correlación de Pearson', fontsize=12)
    plt.ylabel('Variable Predictora', fontsize=12)
    plt.axvline(x=0, color='black', linewidth=0.8)
    plt.tight_layout()

    # --- Guardado en ambos formatos ---
    ruta_png = os.path.join(output_dir, 'eda_correlacion_con_target.png')
    ruta_tex = os.path.join(output_dir, 'eda_correlacion_con_target.tex')
    plt.savefig(ruta_png)
    tikzplotlib.save(ruta_tex)
    logger.info(f"Gráfico de correlación con target guardado en: {ruta_png} y .tex")
    plt.show()

if __name__ == "__main__":
    logger.info("--- Iniciando Análisis Exploratorio de Datos Preliminar ---")
    df_base = data_processing.crear_dataframe_base()
    output_path = os.path.join(config.PATHS['OUTPUTS'], 'preliminary_eda')
    os.makedirs(output_path, exist_ok=True)

    plot_data_completeness(df_base, output_path)
    
    # --- NUEVO: Generar y guardar la tabla de estadísticas descriptivas ---
    logger.info("\nGenerando tabla de estadísticas descriptivas por estación...")
    
    # Nos enfocamos en la variable objetivo y las WQ features más importantes
    vars_para_resumen = [config.TARGET_COLUMN] + config.WATER_QUALITY_FEATURES
    
    resumen_stats = analysis_tools.resumen_estadistico_por_estacion(df_base, variables=vars_para_resumen)
    
    ruta_tabla = os.path.join(output_path, 'eda_estadisticas_descriptivas.csv')
    resumen_stats.to_csv(ruta_tabla)
    
    logger.info(f"Tabla de estadísticas descriptivas guardada en: {ruta_tabla}")
    print("\n--- Resumen Estadístico por Estación (primeras 5 filas) ---")
    print(resumen_stats.head())
    # --- FIN DEL NUEVO CÓDIGO ---
    
    # Para el análisis de correlación, usaremos TODAS las variables predictoras disponibles
    features_completas = config.ALL_PREDICTOR_VARIABLES
    
    logger.info(f"\nGenerando matriz de correlación para todas las variables...")
    analysis_tools.plot_correlation_heatmap(df_base, features_completas, output_path, "preliminary_full")
    
    plot_target_correlation(df_base, features_completas, config.TARGET_COLUMN, output_path)
    
    logger.info("\nGenerando boxplot de distribución de la variable objetivo...")
    analysis_tools.plot_boxplot_por_estacion(
    df_base, 
    config.TARGET_COLUMN, 
    output_path, 
    "preliminary"
    )

    logger.info("--- Análisis Exploratorio Preliminar Finalizado ---")