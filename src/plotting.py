# src/plotting.py
# Modulo para las funciones que generan las visualizaciones.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from matplotlib.patches import Patch
from sklearn.model_selection import KFold, TimeSeriesSplit
from src import config

logger = logging.getLogger('PipelineLogger')

def _configurar_estilo_plot():
    """Establece una configuracion de estilo consistente para todos los graficos."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'figure.figsize': (16, 8), 'axes.titlesize': 18, 'axes.labelsize': 14})

# --- FUNCIONES DE GRAFICOS DE RESULTADOS DE ML ---

def _plot_feature_importance(model, features, output_path, run_id):
    """Grafica la importancia de las variables de un modelo tipo arbol."""
    _configurar_estilo_plot()
    importances = model.feature_importances_
    df_importance = pd.DataFrame({'feature': features, 'importance': importances})
    df_importance = df_importance.sort_values(by='importance', ascending=False).head(15)
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=df_importance, palette='viridis')
    plt.title('Importancia de las Variables (Feature Importance)')
    plt.xlabel('Importancia Relativa'); plt.ylabel('Variable')
    plt.tight_layout()
    filename = f"{run_id}_feature_importance.png"
    plt.savefig(os.path.join(output_path, filename))
    plt.close()
    logger.info(f"  - Grafico de importancia de variables guardado: {filename}")

def _plot_predictions_vs_actual(y_true, y_pred, model_name, output_path, run_id, scale_name=""):
    """Grafica los valores predichos vs. los valores reales."""
    _configurar_estilo_plot()
    plt.figure(figsize=(8, 8))
    sns.regplot(x=y_true, y=y_pred, scatter_kws={'alpha':0.6}, line_kws={'color': 'red', 'linestyle': '--', 'lw': 2})
    title = f'Valores Reales vs. Predichos ({model_name})'
    xlabel, ylabel = 'Valores Reales', 'Valores Predichos'
    if scale_name:
        title += f' {scale_name}'; xlabel += f' {scale_name}'; ylabel += f' {scale_name}'
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.grid(True); plt.tight_layout()
    filename_suffix = f"_preds_vs_actual{'_' + scale_name.lower().replace('(', '').replace(')', '') if scale_name else ''}.png"
    filename = f"{run_id}_{model_name.replace(' ', '_').lower()}{filename_suffix}"
    plt.savefig(os.path.join(output_path, filename))
    plt.close()
    logger.info(f"  - Grafico de predicciones vs. reales guardado: {filename}")

def generar_plots_de_modelo(model, fold_data, plot_output_path, exp_config, run_id):
    """Genera y guarda todos los graficos de evaluacion para un modelo entrenado."""
    model_name = exp_config['model_name']
    y_test_log = fold_data['y_test']; preds_log = fold_data['predictions_log']; features = fold_data['features']
    
    _plot_predictions_vs_actual(y_test_log, preds_log, model_name, plot_output_path, run_id, scale_name="(Escala Log)")
    
    if hasattr(model, 'feature_importances_'):
        _plot_feature_importance(model, features, plot_output_path, run_id)

# --- FUNCIONES DE GRAFICOS DE ANALISIS EXPLORATORIO (EDA) ---

def plot_serie_temporal_con_outliers(df, variable, output_folder, exp_config, run_id):
    """Genera un grafico de dispersion de serie temporal, destacando outliers."""
    logger.info(f"  - Generando grafico de diagnostico para '{variable}'...")
    _configurar_estilo_plot()
    df_plot = df[[config.DATE_COLUMN, variable]].copy().dropna(subset=[variable])
    if df_plot.empty: return
    df_plot[config.DATE_COLUMN] = pd.to_datetime(df_plot[config.DATE_COLUMN])
    df_plot = df_plot.set_index(config.DATE_COLUMN).sort_index()

    outlier_params = exp_config.get('outlier_removal_params', {}); iqr_multiplier = outlier_params.get('iqr_multiplier', 1.5)
    data_to_check = np.log1p(df_plot[variable]) if variable == config.TARGET_COLUMN else df_plot[variable]
    Q1, Q3 = data_to_check.quantile(0.25), data_to_check.quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior, limite_superior = Q1 - iqr_multiplier * IQR, Q3 + iqr_multiplier * IQR
    if variable == config.TARGET_COLUMN: limite_inferior, limite_superior = np.expm1(limite_inferior), np.expm1(limite_superior)
    
    df_plot['is_outlier'] = (df_plot[variable] < limite_inferior) | (df_plot[variable] > limite_superior)
    
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.scatter(df_plot[~df_plot['is_outlier']].index, df_plot[~df_plot['is_outlier']][variable], marker='+', color='black', s=50, linewidth=0.8, label='Valores Normales')
    if df_plot['is_outlier'].any(): ax.scatter(df_plot[df_plot['is_outlier']].index, df_plot[df_plot['is_outlier']][variable], marker='x', color='red', s=60, linewidth=1, label='Outliers Detectados')
    ax.axhspan(ax.get_ylim()[0], limite_inferior, facecolor='none', edgecolor='grey', alpha=0.5, hatch='///'); ax.axhspan(limite_superior, ax.get_ylim()[1], facecolor='none', edgecolor='grey', alpha=0.5, hatch='///')
    
    title = f'Serie Temporal y Outliers para {variable.replace("_", " ").title()}'
    if variable == config.TARGET_COLUMN: ax.set_yscale('log'); title += ' (Escala Logaritmica)'
    ax.set_title(title, fontsize=18); ax.set_xlabel('Fecha'); ax.set_ylabel('Valor Medido')
    ax.legend(); plt.tight_layout()
    filename = f"{run_id}_eda_outliers_{variable}.png"
    plt.savefig(os.path.join(output_folder, filename))
    plt.close(fig)

# Dentro de src/plotting.py

def plot_predictions_over_time(y_true_log, y_pred_log, dates, model_name, output_path, run_id):
    """
    Grafica las predicciones del modelo y los valores reales a lo largo del tiempo
    utilizando un gráfico de dispersión para evitar líneas que unan los puntos.
    """
    logger.info("Generando grafico de predicciones en el tiempo...")
    _configurar_estilo_plot()

    df_plot = pd.DataFrame({
        'fecha': dates.values,
        'Valores Reales (log)': y_true_log.values,
        'Predicciones del Modelo (log)': y_pred_log
    })
    df_plot = df_plot.sort_values('fecha')

    plt.figure(figsize=(20, 8))
    
    # --- CAMBIO PRINCIPAL: Usamos scatter en lugar de plot ---
    # Graficamos los valores reales como círculos azules
    plt.scatter(df_plot['fecha'], df_plot['Valores Reales (log)'], 
                label='Valores Reales', color='royalblue', marker='o', s=50, alpha=0.7)
    
    # Graficamos las predicciones como cruces rojas
    plt.scatter(df_plot['fecha'], df_plot['Predicciones del Modelo (log)'], 
                label='Predicciones del Modelo', color='red', marker='x', s=50, alpha=0.8)
    # --- FIN DEL CAMBIO ---
    
    plt.title(f'Comparacion de Predicciones en el Tiempo - {model_name} (Escala Log)', fontsize=18)
    plt.xlabel('Fecha')
    plt.ylabel('Coliformes Fecales (escala log)')
    plt.legend()
    plt.grid(True) # Añadimos una grilla para mejor legibilidad
    plt.tight_layout()
    
    filename = f"{run_id}_{model_name.replace(' ', '_').lower()}_predictions_over_time.png"
    plt.savefig(os.path.join(output_path, filename))
    plt.close()
    logger.info(f"Grafico de predicciones en el tiempo guardado en {filename}")