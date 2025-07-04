# main.py
"""
Orquestador Principal de la Plataforma de Experimentación de Calidad de Agua.

Este script es el punto de entrada para ejecutar el pipeline completo, incluyendo
la validación automática post-optimización.
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importamos nuestros módulos
from src import config, utils, logger_config, train, data_processing, clustering, plotting, analysis_tools

logger = logging.getLogger('PipelineLogger')


def _generar_reporte_comparativo_final(all_results_list):
    """
    Toma una lista de resultados, los consolida y genera artefactos comparativos.
    """
    if not all_results_list:
        logger.warning("No se generaron resultados para comparar.")
        return

    df_completo = pd.concat(all_results_list, ignore_index=True)
    df_completo['cluster_id'] = df_completo['cluster_id'].fillna('General')
    
    ruta_tabla_completa = os.path.join(config.PATHS['OUTPUTS'], 'comparacion_detallada_experimentos.csv')
    df_completo.to_csv(ruta_tabla_completa, index=False)
    logger.info(f"Tabla detallada de todos los folds y experimentos guardada en: {ruta_tabla_completa}")

    metricas_resumen = [col for col in ['R2_log', 'RMSE_log', 'PBIAS_log', 'NSE_log'] if col in df_completo.columns]
    summary_final = df_completo.groupby(['base_experiment_id', 'cluster_id'])[metricas_resumen].mean().round(4)
    logger.info("\n--- TABLA RESUMEN COMPARATIVA FINAL ---")
    print(summary_final.sort_values(by='R2_log', ascending=False).to_string())
    
    ruta_tabla_resumen = os.path.join(config.PATHS['OUTPUTS'], 'comparacion_final_experimentos.csv')
    summary_final.to_csv(ruta_tabla_resumen)
    logger.info(f"\nTabla resumen comparativa guardada en: {ruta_tabla_resumen}")

    df_plot = summary_final.reset_index().sort_values('R2_log', ascending=False)
    df_plot['display_id'] = df_plot['base_experiment_id'] + " (" + df_plot['cluster_id'].astype(str) + ")"
    df_plot['run_type'] = df_plot['cluster_id'].apply(lambda x: 'Por Cluster' if x != 'General' else 'General')
    
    plt.figure(figsize=(16, max(8, len(df_plot) * 0.5)))
    sns.barplot(x='R2_log', y='display_id', data=df_plot, hue='run_type', palette='viridis', orient='h', dodge=False)
    plt.title(r'Comparacion de Rendimiento ($R^2$ Log) entre Experimentos', fontsize=18)
    plt.xlabel(r'$R^2$ (log) Promedio (Barras mas largas son mejores)', fontsize=14)
    plt.ylabel('ID del Experimento (y Cluster)', fontsize=14)
    plt.legend(title='Tipo de Corrida')
    plt.tight_layout()
    
    ruta_grafico = os.path.join(config.PATHS['OUTPUTS'], 'comparison_chart.png')
    plt.savefig(ruta_grafico)
    logger.info(f"Grafico comparativo final guardado en: {ruta_grafico}")
    plt.show()


def ejecutar_un_experimento(df_base, exp_config, cluster_id=None):
    """
    Ejecuta un pipeline completo para un único experimento y devuelve sus resultados y
    los mejores parámetros encontrados (si es un experimento de optimización).
    """
    run_id = exp_config['id']
    run_output_dir = os.path.join(config.PATHS['OUTPUTS'], run_id)
    os.makedirs(os.path.join(run_output_dir, "models"), exist_ok=True)
    plot_output_path = os.path.join(run_output_dir, "plots")
    os.makedirs(plot_output_path, exist_ok=True)
    
    log_file_path = os.path.join(run_output_dir, 'run_report.log')
    logger_config.setup_logger(log_file_path)
    
    logger.info(f"--- INICIANDO EXPERIMENTO: {run_id} ---")

    df_filtrado, _, features_finales = data_processing.filtrar_por_casos_completos(df_base.copy(), exp_config)
    
    if exp_config.get('generate_eda_plots', False):
        eda_plot_path = os.path.join(plot_output_path, "EDA")
        os.makedirs(eda_plot_path, exist_ok=True)
        # ... (lógica para plots de EDA)
        
    if df_filtrado.shape[0] < 20:
        logger.error(f"DataFrame para '{run_id}' con muy pocas filas. Abortando.")
        return None, None

    cols_a_guardar = config.METADATA_COLUMNS + [config.TARGET_COLUMN] + exp_config['feature_set']
    df_filtrado[[col for col in cols_a_guardar if col in df_filtrado.columns]].to_csv(os.path.join(run_output_dir, 'datos_usados_en_modelo.csv'), index=False)
    
    final_model, summary_df, results_df, last_fold_data, best_params = train.ejecutar_experimento_ml(
        df_filtrado, exp_config, run_output_dir, cluster_id=cluster_id
    )
    
    if final_model and last_fold_data:
        plotting.generar_plots_de_modelo(final_model, last_fold_data, plot_output_path, exp_config, run_id)
        test_dates = df_filtrado.loc[last_fold_data['test_indices'], config.DATE_COLUMN]
        plotting.plot_predictions_over_time(last_fold_data['y_test'], last_fold_data['predictions_log'], test_dates, exp_config['model_name'], plot_output_path, run_id)
        analysis_tools.generar_reporte_rendimiento_espacial(df_filtrado, last_fold_data, run_output_dir, run_id)
    
    logger.info(f"--- EXPERIMENTO {run_id} FINALIZADO ---")
    return results_df, best_params

# --- FUNCIÓN main() DEFINITIVA Y COMPLETA ---
def main():
    """
    Función principal que orquesta la ejecución de todos los experimentos.
    """
    logger.info("==================================================")
    logger.info("INICIANDO PLATAFORMA DE EXPERIMENTACION")
    
    df_base = data_processing.crear_dataframe_base()
    df_cuencas = pd.read_csv(config.PATHS['INPUT_DATA_CUENCAS'])
    all_experiment_results = []
    
    logger.info("\n--- Iniciando Bucle Principal de Experimentos ---")
    for exp_config in config.EXPERIMENTS_TO_RUN:
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_config['original_id'] = exp_config['id'] # Guardamos el ID base para el reporte

        clustering_params = exp_config.get('clustering_params', {'apply': False})
        
        # --- LÓGICA DE CLUSTERING REINTEGRADA ---
        if clustering_params.get('apply'):
            # --- CASO 1: EXPERIMENTO POR CLÚSTER ---
            k = clustering_params['n_clusters']
            logger.info(f"\n--- INICIANDO EXPERIMENTO POR CLUSTERS: {exp_config['original_id']} (k={k}) ---")
            
            df_station_clusters = clustering.get_station_clusters(df_cuencas, k=k, feature_set=config.BASIN_FEATURES)
            df_con_clusters = pd.merge(df_base, df_station_clusters, on='station_code', how='left')

            for cluster_id in sorted(df_con_clusters['cluster_id'].dropna().unique()):
                logger.info(f"  --- > Procesando para Cluster ID: {cluster_id} ---")
                
                df_para_experimento_cluster = df_con_clusters[df_con_clusters['cluster_id'] == cluster_id]
                
                exp_config_cluster = exp_config.copy()
                exp_config_cluster['id'] = f"{timestamp}_{exp_config['original_id']}_cluster{int(cluster_id)}"
                
                # Ejecutamos el pipeline para este subconjunto de datos
                results, best_params = ejecutar_un_experimento(df_para_experimento_cluster, exp_config_cluster, cluster_id=int(cluster_id))
                if results is not None:
                    all_experiment_results.append(results)
                
                # Aquí no se realiza validación post-optimización para no complicar el flujo,
                # pero se podría añadir si fuera necesario.

        else:
            # --- CASO 2: EXPERIMENTO GENERAL (SIN CLUSTERING) ---
            exp_config['id'] = f"{timestamp}_{exp_config['original_id']}"
            logger.info(f"\n--- INICIANDO EXPERIMENTO ÚNICO: {exp_config['id']} ---")
            
            results, best_params_encontrados = ejecutar_un_experimento(df_base, exp_config)
            if results is not None:
                all_experiment_results.append(results)

            if exp_config.get('optimization_strategy') and best_params_encontrados:
                logger.info(f"\n--- INICIANDO RUN DE VALIDACIÓN AUTOMÁTICA para {exp_config['original_id']} ---")
                
                validation_config = exp_config.copy()
                validation_config['id'] = f"{exp_config['id']}_VALIDATION"
                validation_config['original_id'] = f"{exp_config['original_id']}_VALIDATION"
                validation_config.pop('optimization_strategy', None)
                validation_config.pop('param_grid', None)
                validation_config.pop('search_space', None)
                validation_config['hyperparameters'] = best_params_encontrados
                
                validation_results, _ = ejecutar_un_experimento(df_base, validation_config)
                if validation_results is not None:
                    all_experiment_results.append(validation_results)

    # --- REPORTE FINAL ---
    if config.REPORTING_PARAMS.get('generate_final_comparison_report', False) and all_experiment_results:
        logger.info("\n--- Generando Reporte Comparativo Final ---")
        _generar_reporte_comparativo_final(all_experiment_results)

    logger.info("\n==================================================")
    logger.info("TODOS LOS EXPERIMENTOS HAN FINALIZADO.")
    logger.info("==================================================")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
    utils.setup_directorios()
    main()