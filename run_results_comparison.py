# run_results_comparison.py
# Script para analizar y comparar los resultados de multiples experimentos.

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import sys

# Importamos la configuracion para saber donde buscar
from src import config, utils

# Configurar un logger basico para ver los mensajes en la consola.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)


def comparar_resultados_experimentos():
    """
    Busca recursivamente todos los archivos de resultados en la carpeta de outputs,
    los une y genera un grafico y una tabla comparativa.
    """
    output_path = config.PATHS['OUTPUTS']
    all_results = []

    logging.info(f"Buscando archivos 'summary_results.csv' dentro de: {output_path}")

    # 1. Busqueda recursiva de archivos de resultados
    for root, dirs, files in os.walk(output_path):
        if 'summary_results.csv' in files:
            run_id = os.path.basename(root)
            file_path = os.path.join(root, 'summary_results.csv')
            
            try:
                # Leemos el CSV y procesamos el encabezado complejo
                df_res = pd.read_csv(file_path, index_col=0, header=[0, 1])
                
                # Aplanamos el MultiIndex de las columnas
                df_res.columns = ['_'.join(col).strip() for col in df_res.columns.values]
                df_res = df_res.reset_index().rename(columns={'index': 'model'})

                df_res['experiment_id'] = run_id
                all_results.append(df_res)
                logging.info(f"  - Resultado encontrado y cargado para la corrida: {run_id}")
            except Exception as e:
                logging.error(f"Error cargando o procesando el archivo {file_path}: {e}")

    if not all_results:
        logging.warning("No se encontraron archivos de resultados para comparar. Ejecuta primero 'main.py'.")
        return
        
    # 2. Consolidar todos los resultados en un unico DataFrame
    df_comparativo = pd.concat(all_results, ignore_index=True)
    
    # 3. Mostrar y GUARDAR la tabla comparativa
    logging.info("\n--- Tabla Comparativa de Resultados ---")
    df_display = df_comparativo.sort_values('R2_log_mean', ascending=False)
    print(df_display[['experiment_id', 'model', 'R2_log_mean', 'RMSE_log_mean', 'training_time_secs_mean']].round(4))
    
    # --- NUEVO: Guardar la tabla resumen en un archivo CSV ---
    ruta_tabla_resumen = os.path.join(output_path, 'comparacion_final_experimentos.csv')
    df_display.to_csv(ruta_tabla_resumen, index=False)
    logging.info(f"\nTabla resumen comparativa guardada en: {ruta_tabla_resumen}")
    # --- FIN DEL NUEVO CODIGO ---

    # 4. Generar y guardar el grafico comparativo
    plt.style.use('ggplot')
    df_plot = df_display
    
    plt.figure(figsize=(14, max(6, len(df_plot) * 0.8)))
    # Usamos la sintaxis de seaborn más reciente para evitar el warning
    sns.barplot(x='R2_log_mean', y='experiment_id', data=df_plot, hue='experiment_id', palette='viridis', orient='h', legend=False)
    
    # --- CORRECCIÓN: Usar notación LaTeX para el superíndice 2 ---
    plt.title(r'Comparacion de Rendimiento ($R^2$ Log) entre Experimentos', fontsize=16)
    plt.xlabel(r'$R^2$ (log) Promedio (Barras mas largas son mejores)', fontsize=12)
    plt.ylabel('ID del Experimento', fontsize=12)
    plt.tight_layout()
    
    # Guardar el grafico en la carpeta principal de outputs
    ruta_grafico = os.path.join(output_path, 'comparison_chart.png')
    plt.savefig(ruta_grafico)
    logging.info(f"\nGrafico comparativo guardado en: {ruta_grafico}")
    plt.show()


if __name__ == "__main__":
    # Asegurarnos de que la carpeta de outputs exista antes de intentar guardar algo
    utils.setup_directorios()
    comparar_resultados_experimentos()