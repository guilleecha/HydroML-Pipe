# run_validation_plots.py
# Script para generar visualizaciones de las estrategias de validacion cruzada.

import pandas as pd
import os
import logging
import sys
import json

# Importamos desde nuestra nueva estructura 'src'
from src import config, utils, analysis_tools

# Configurar un logger basico para ver los mensajes en la consola.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger('PipelineLogger')

def main():
    """
    Funcion principal que encuentra todas las estrategias de validacion unicas
    y genera un grafico representativo para cada una.
    """
    logger.info("--- INICIANDO GENERACION DE GRAFICOS DE VALIDACION CRUZADA ---")
    
    output_path = config.PATHS['OUTPUTS']
    
    # 1. Encontrar todos los experimentos y sus estrategias
    experimentos = []
    for folder_name in os.listdir(output_path):
        run_output_dir = os.path.join(output_path, folder_name)
        if os.path.isdir(run_output_dir):
            # Asumimos que la configuracion se puede deducir del nombre o de un archivo guardado
            # Por simplicidad, aquí deducimos del nombre del experimento en config.py
            for exp_config in config.EXPERIMENTS_TO_RUN:
                if exp_config['id'] in folder_name:
                    experimentos.append({
                        'run_id': folder_name,
                        'strategy': exp_config['validation_strategy'],
                        'data_path': os.path.join(run_output_dir, 'datos_usados_en_modelo.csv')
                    })
                    break

    if not experimentos:
        logger.warning("No se encontraron carpetas de experimentos en 'outputs/'.")
        return

    # 2. Obtener las estrategias unicas que se han ejecutado
    estrategias_unicas = {exp['strategy'] for exp in experimentos}
    logger.info(f"Estrategias de validacion encontradas: {list(estrategias_unicas)}")

    # 3. Para cada estrategia unica, generar UN grafico representativo
    for strategy in estrategias_unicas:
        # Encontrar el primer experimento que uso esta estrategia
        exp_representativo = next((exp for exp in experimentos if exp['strategy'] == strategy), None)
        
        if exp_representativo:
            logger.info(f"\nGenerando grafico para la estrategia '{strategy}' usando la corrida '{exp_representativo['run_id']}' como ejemplo...")
            
            try:
                df_experimento = pd.read_csv(exp_representativo['data_path'])
                df_experimento[config.DATE_COLUMN] = pd.to_datetime(df_experimento[config.DATE_COLUMN])
                df_experimento = df_experimento.sort_values(by=config.DATE_COLUMN)

                # Preparar X e y
                target = config.TARGET_COLUMN
                features = [f for f in df_experimento.columns if f not in config.METADATA_COLUMNS + [target]]
                X = df_experimento[features]
                y = df_experimento[target]

                # Generar el plot
                plot_output_path = os.path.join(config.PATHS['OUTPUTS'], 'visualizations', 'eda')
                os.makedirs(plot_output_path, exist_ok=True)
                analysis_tools.plot_cv_splits(strategy, X, y, plot_output_path)

            except FileNotFoundError:
                logger.error(f"No se pudo encontrar el archivo de datos para la corrida: {exp_representativo['run_id']}")
            except Exception as e:
                logger.error(f"Ocurrio un error al procesar la corrida {exp_representativo['run_id']}: {e}")

    logger.info("\n✅ Proceso de generacion de graficos de validacion finalizado.")


if __name__ == "__main__":
    utils.setup_directorios()
    main()