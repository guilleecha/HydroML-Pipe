# scripts/generar_tabla_latex.py

import pandas as pd
import sys
import os
import logging

# --- Setup ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from src import config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)


def generar_codigo_latex_tabla():
    """
    Lee el archivo de resultados finales y genera el código LaTeX para la tabla resumen.
    """
    ruta_resultados = os.path.join(config.PATHS['OUTPUTS'], 'comparacion_final_experimentos.csv')
    try:
        df = pd.read_csv(ruta_resultados)
    except FileNotFoundError:
        logger.error(f"No se encontró el archivo de resultados en: {ruta_resultados}")
        logger.error("Por favor, ejecuta 'main.py' primero para generar los resultados.")
        return

    # Preparamos los datos para la tabla
    df = df.sort_values(by=['base_experiment_id', 'cluster_id']).reset_index(drop=True)
    df['base_experiment_id'] = df['base_experiment_id'].str.replace('_', ' ', regex=False)

    # --- INICIO DE LA GENERACIÓN DE CÓDIGO LATEX ---
    
    print("\n" + "%" * 50)
    print("% COPIA Y PEGA EL SIGUIENTE CÓDIGO EN TU DOCUMENTO LATEX")
    print("%" * 50 + "\n")
    
    print(r'\begin{table}[H]')
    print(r'    \caption{Resumen comparativo del rendimiento de los experimentos finales, agrupados por configuración base y subgrupo (clúster).}')
    print(r'    \label{tab:resultados_finales}')
    print(r'    \centering')
    # Usamos tabular en lugar de tabularx para un control más simple del ancho
    print(r'    \begin{tabular}{l l c c c}')
    print(r'        \toprule')
    print(r'        \textbf{Experimento Base} & \textbf{Subgrupo} & \textbf{$R^2$ (log)} & \textbf{RMSE (log)} & \textbf{PBIAS (log)} \\')
    print(r'        \midrule')

    last_base_id = None
    for i, row in df.iterrows():
        # Para la primera fila de un nuevo grupo, usamos \multirow
        if row['base_experiment_id'] != last_base_id:
            if last_base_id is not None:
                print(r'        \midrule') # Línea divisoria entre grupos
            
            # Contamos cuántas filas ocupa este grupo
            group_size = len(df[df['base_experiment_id'] == row['base_experiment_id']])
            
            # Creamos la primera parte de la fila con \multirow
            # El * en multirow significa que el texto se ajustará automáticamente en ancho
            id_cell = f"        \\multirow{{{group_size}}}{{*}}{{{row['base_experiment_id']}}}"
        else:
            # Para las otras filas del mismo grupo, la primera celda va vacía
            id_cell = "        "
            
        # Creamos el resto de la fila
        cluster_cell = str(row['cluster_id'])
        r2_cell = f"{row['R2_log']:.3f}" if pd.notna(row['R2_log']) else '---'
        rmse_cell = f"{row['RMSE_log']:.3f}" if pd.notna(row['RMSE_log']) else '---'
        pbias_cell = f"{row.get('PBIAS_log', 'nan'):.2f}" if pd.notna(row.get('PBIAS_log')) else '---'
        
        print(f"{id_cell} & {cluster_cell} & {r2_cell} & {rmse_cell} & {pbias_cell} \\\\")
        
        last_base_id = row['base_experiment_id']

    print(r'        \bottomrule')
    print(r'    \end{tabular}')
    print(r'\end{table}')
    print("\n" + "%" * 50)


if __name__ == "__main__":
    generar_codigo_latex_tabla()