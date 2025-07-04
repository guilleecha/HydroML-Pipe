# src/analysis_tools.py
# Modulo con funciones para Analisis Exploratorio de Datos (EDA),
# tanto numerico como visual.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from matplotlib.patches import Patch
from sklearn.model_selection import KFold, TimeSeriesSplit
from datetime import datetime
import tikzplotlib 
from sklearn.metrics import r2_score, mean_squared_error



from src import config

# Obtenemos el logger que ya fue configurado
logger = logging.getLogger('PipelineLogger')

# =============================================================================
# SECCION 1: FUNCIONES DE ANALISIS ESTADISTICO
# =============================================================================

def resumen_estadistico_por_estacion(df, variables=None):
    """
    Calcula un resumen estadístico por estación para las variables especificadas,
    incluyendo cuartiles.
    """
    logger.info("Generando resumen estadistico por estacion...")
    if variables is None:
        variables = df.select_dtypes(include='number').columns.tolist()

    variables_existentes = [v for v in variables if v in df.columns]
    
    if not variables_existentes:
        logger.error("Ninguna de las variables especificadas existe en el DataFrame.")
        return pd.DataFrame()

    for var in variables_existentes:
        df[var] = pd.to_numeric(df[var], errors='coerce')

    # --- MEJORA: Añadimos Q1 y Q3 al cálculo ---
    resumen = (df
               .groupby('station_code')[variables_existentes]
               .agg([
                   ('N', 'count'),
                   ('Media', 'mean'),
                   ('StdDev', 'std'),
                   ('Min', 'min'),
                   ('Q1_25%', lambda x: x.quantile(0.25)),
                   ('Mediana_50%', 'median'),
                   ('Q3_75%', lambda x: x.quantile(0.75)),
                   ('Max', 'max')
               ])
               .round(2))

    return resumen


def frecuencia_muestreo_por_estacion(df):
    """
    Analiza la frecuencia de muestreo (en dias) por estacion.
    """
    logger.info("Analizando frecuencia de muestreo por estacion...")
    resultados = []
    df_temp = df.copy()
    df_temp[config.DATE_COLUMN] = pd.to_datetime(df_temp[config.DATE_COLUMN])

    for station, group in df_temp.groupby('station_code'):
        fechas = group[config.DATE_COLUMN].dropna().sort_values()
        n = len(fechas)

        if n < 2:
            resultados.append({
                'station_code': station, 'n_muestras': n, 'frecuencia_media_dias': None,
                'frecuencia_min_dias': None, 'frecuencia_max_dias': None
            })
            continue

        difs = fechas.diff().dt.days.dropna()
        resultados.append({
            'station_code': station, 'n_muestras': n, 'frecuencia_media_dias': difs.mean(),
            'frecuencia_min_dias': difs.min(), 'frecuencia_max_dias': difs.max()
        })

    return pd.DataFrame(resultados).round(2)


# =============================================================================
# SECCION 2: FUNCIONES DE ANALISIS VISUAL (GRAFICOS)
# =============================================================================

# --- FUNCIONES DE ANALISIS VISUAL (GRAFICOS) ---

def _configurar_estilo_plot_eda():
    """Establece una configuracion de estilo consistente para los graficos de EDA."""
    plt.style.use('ggplot')
    plt.rcParams.update({'figure.figsize': (16, 8), 'axes.titlesize': 18, 'axes.labelsize': 14})


def plot_serie_temporal_con_outliers(df, variable, output_folder, exp_config, run_id):
    """
    Genera un grafico de dispersion de serie temporal.
    Si el tratamiento de outliers está activado en la configuración, los destaca.
    De lo contrario, grafica todos los puntos de manera uniforme.
    """
    logger.info(f"  - Generando grafico de diagnostico de outliers para '{variable}'...")
    
    # 1. Configurar estilo y preparar datos
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'figure.figsize': (20, 8)})
    
    df_plot = df[[config.DATE_COLUMN, variable]].copy().dropna(subset=[variable])
    df_plot[config.DATE_COLUMN] = pd.to_datetime(df_plot[config.DATE_COLUMN])
    df_plot = df_plot.set_index(config.DATE_COLUMN).sort_index()

    if df_plot.empty: return

    # 2. Creacion del grafico
    fig, ax = plt.subplots()

    # --- NUEVA LÓGICA CONDICIONAL ---
    outlier_params = exp_config.get('outlier_removal_params', {})
    apply_treatment = outlier_params.get('apply', False)

    if apply_treatment:
        # --- COMPORTAMIENTO SI SE APLICAN OUTLIERS (como antes) ---
        iqr_multiplier = outlier_params.get('iqr_multiplier', 1.5)
        data_to_check = np.log1p(df_plot[variable]) if variable == config.TARGET_COLUMN else df_plot[variable]
        Q1, Q3 = data_to_check.quantile(0.25), data_to_check.quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior, limite_superior = Q1 - iqr_multiplier * IQR, Q3 + iqr_multiplier * IQR
        
        if variable == config.TARGET_COLUMN:
            limite_inferior, limite_superior = np.expm1(limite_inferior), np.expm1(limite_superior)
        
        df_plot['is_outlier'] = (df_plot[variable] < limite_inferior) | (df_plot[variable] > limite_superior)
        df_inliers = df_plot[~df_plot['is_outlier']]
        df_outliers = df_plot[df_plot['is_outlier']]

        ax.scatter(df_inliers.index, df_inliers[variable], marker='+', color='black', s=50, linewidth=0.8, label='Valores Normales')
        if not df_outliers.empty:
            ax.scatter(df_outliers.index, df_outliers[variable], marker='x', color='red', s=60, linewidth=1, label='Outliers Detectados')

        ax.axhspan(ax.get_ylim()[0], limite_inferior, facecolor='none', edgecolor='grey', alpha=0.5, hatch='///', label='Zona de Outliers')
        ax.axhspan(limite_superior, ax.get_ylim()[1], facecolor='none', edgecolor='grey', alpha=0.5, hatch='///')
        
        title_suffix = 'y Deteccion de Outliers'
    else:
        # --- COMPORTAMIENTO SI NO SE APLICAN OUTLIERS (nuevo) ---
        ax.scatter(df_plot.index, df_plot[variable], marker='+', color='black', s=50, linewidth=0.8, label='Todos los Valores')
        title_suffix = '(Sin Tratamiento de Outliers)'

    # 3. Trazar lineas de media y mediana (siempre útil)
    mean_val, median_val = df_plot[variable].mean(), df_plot[variable].median()
    ax.axhline(mean_val, color='black', linestyle='--', lw=1.5, label=f'Media: {mean_val:.2f}')
    ax.axhline(median_val, color='dimgray', linestyle=':', lw=1.5, label=f'Mediana: {median_val:.2f}')

    # 4. Configurar titulos y etiquetas
    title = f'Serie Temporal para {variable.replace("_", " ").title()} {title_suffix}'
    if variable == config.TARGET_COLUMN:
        ax.set_yscale('log'); title += ' (Escala Logaritmica)'
        
    ax.set_title(title, fontsize=18); ax.set_xlabel('Fecha'); ax.set_ylabel('Valor Medido')
    ax.legend(); plt.tight_layout()
    
    # 5. Guardar
    filename = f"{run_id}_eda_outliers_{variable}.png"
    plt.savefig(os.path.join(output_folder, filename))
    plt.close(fig)




def plot_boxplot_por_estacion(df, variable, output_folder, run_id):
    """
    Genera un boxplot de una variable por estación, agrupando por cuenca y
    ordenando por área. El gráfico se guarda en formato PNG y TikZ con
    un estilo en blanco y negro para el informe.
    """
    logger.info(f"Generando boxplot por estación para la variable '{variable}'...")
    
    # 1. Preparar los datos
    df_plot = df[['station_code', 'basin', 'basin_area_a', variable]].copy().dropna(subset=[variable])
    if df_plot.empty:
        logger.warning(f"No hay datos para generar el boxplot de '{variable}'.")
        return

    # Usamos escala logarítmica para coliformes, como en el modelo
    if variable == config.TARGET_COLUMN:
        df_plot[variable] = np.log1p(df_plot[variable])
        y_label = f'{variable}_log'
    else:
        y_label = variable

    # 2. Ordenar las estaciones por cuenca y luego por área
    orden_estaciones = (
        df_plot.drop_duplicates('station_code')
        .sort_values(['basin', 'basin_area_a'])['station_code']
    )
    
    # 3. Preparar información para las divisiones y etiquetas
    info_est = df_plot[['station_code', 'basin']].drop_duplicates().set_index('station_code').loc[orden_estaciones]
    cambios_cuenca = info_est['basin'] != info_est['basin'].shift()
    indices_separadores = [i for i, val in enumerate(cambios_cuenca) if val][1:]

    # 4. Crear el gráfico en blanco y negro
    plt.style.use('grayscale')
    plt.figure(figsize=(20, 8))
    ax = sns.boxplot(
        data=df_plot, x='station_code', y=variable, order=orden_estaciones,
        color='white', linecolor='black', linewidth=0.75,
        medianprops={'color': 'black', 'linewidth': 2},
        whiskerprops={'color': 'black', 'linestyle': '--'},
        capprops={'color': 'black'}
    )

    # Añadir las líneas divisorias
    for pos in indices_separadores:
        plt.axvline(pos - 0.5, color='black', linestyle='--', alpha=0.7)

    # 5. Añadir Títulos y Etiquetas
    plt.title(f'Distribución de {y_label.replace("_", " ").title()} por Estación', fontsize=18)
    plt.xlabel("Estación (Agrupada por Cuenca)", fontsize=14)
    plt.ylabel(f'Valor ({y_label})', fontsize=14)
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # 6. Guardar en ambos formatos
    filename_base = f"{run_id}_eda_boxplot_{variable}"
    ruta_png = os.path.join(output_folder, f"{filename_base}.png")
    ruta_tex = os.path.join(output_folder, f"{filename_base}.tex")
    plt.savefig(ruta_png)
    tikzplotlib.save(ruta_tex)
    plt.close()
    
    logger.info(f"  - Boxplot por estación guardado en: {ruta_png} y .tex")

def plot_correlation_heatmap(df, feature_list, output_folder, run_id):
    """
    Calcula y grafica un mapa de calor de correlación triangular,
    guardando en PNG y TikZ.
    """
    _configurar_estilo_plot_eda()
    
    logger.info("Generando mapa de calor de correlacion...")
    numeric_features = [f for f in feature_list if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    
    if not numeric_features:
        logger.warning("No se encontraron variables numericas para el mapa de calor.")
        return
        
    df_corr = df[numeric_features].corr()
    
    # Crear una máscara para el triángulo superior
    mask = np.triu(np.ones_like(df_corr, dtype=bool))

    plt.figure(figsize=(20, 16)) # Aumentamos el tamaño para mejor legibilidad
    
    sns.heatmap(df_corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 8})
    
    plt.title('Mapa de Calor de Correlacion de Variables', fontsize=18)
    plt.tight_layout()
    
    # Guardar en ambos formatos
    filename_base = f"{run_id}_eda_correlation_heatmap"
    ruta_png = os.path.join(output_folder, f"{filename_base}.png")
    ruta_tex = os.path.join(output_folder, f"{filename_base}.tex")
    
    plt.savefig(ruta_png)
    tikzplotlib.save(ruta_tex) # Guardamos el código TikZ
    plt.close()
    
    logger.info(f"  - Mapa de calor de correlacion guardado en: {ruta_png} y .tex")

def plot_cv_splits(cv_object, X, y, output_path, run_id, strategy_name):
    """Genera una visualizacion 2D de las divisiones de un objeto CV."""
    logger.info(f"  - Generando grafico de division de CV para '{strategy_name}'...")
    _configurar_estilo_plot_eda()
    fig, ax = plt.subplots(figsize=(15, 8))

    for i, (train, test) in enumerate(cv_object.split(X, y)):
        ax.scatter(train, np.full_like(train, i), marker='_', s=100, lw=10, color='royalblue')
        ax.scatter(test, np.full_like(test, i), marker='_', s=100, lw=10, color='orange')

    ax.set_yticks(np.arange(cv_object.get_n_splits()))
    ax.set_yticklabels([f"Fold {i+1}" for i in range(cv_object.get_n_splits())])
    
    ax.set_ylabel("Iteracion de CV"); ax.set_xlabel("Indice de Muestras (Ordenadas por Fecha)")
    ax.set_title(f"Visualizacion de la Estrategia de Validacion Cruzada: {strategy_name}", fontsize=16)
    ax.legend([Patch(facecolor='royalblue'), Patch(facecolor='orange')], ['Datos de Entrenamiento', 'Datos de Prueba'])
    plt.tight_layout()
    
    filename = f"{run_id}_cv_split_visualization.png"
    plt.savefig(os.path.join(output_path, filename))
    plt.close(fig)
    logger.info(f"  - Grafico de division de CV '{strategy_name}' guardado.")
    
    
# =============================================================================
# SECCION 3: METRICAS RESULTADOS
# =============================================================================




# --- FUNCIONES DE CÁLCULO DE MÉTRICAS ---

def nse(y_true, y_pred):
    """Calcula el Coeficiente de Eficiencia de Nash-Sutcliffe (NSE)."""
    return 1 - (np.sum((y_pred.values - y_true.values)**2) / np.sum((y_true.values - np.mean(y_true.values))**2))

def pbias(y_true, y_pred):
    """Calcula el Percent BIAS (PBIAS)."""
    return 100 * (np.sum(y_pred.values - y_true.values) / np.sum(y_true.values))

def calcular_metricas_grupo(grupo):
    """Calcula un conjunto de métricas para un grupo de datos (DataFrame)."""
    if len(grupo) < 2: # No se pueden calcular métricas con menos de 2 puntos
        return None
    return pd.Series({
        'R2': r2_score(grupo['y_test'], grupo['predictions_log']),
        'NSE': nse(grupo['y_test'], grupo['predictions_log']),
        'RMSE': np.sqrt(mean_squared_error(grupo['y_test'], grupo['predictions_log'])),
        'PBIAS': pbias(grupo['y_test'], grupo['predictions_log']),
        'N_Muestras': len(grupo)
    })


# --- FUNCIÓN PRINCIPAL DE REPORTE ESPACIAL ---

def generar_reporte_rendimiento_espacial(df_datos_original, last_fold_data, output_path, run_id):
    """
    Evalúa el rendimiento del modelo por cuenca y por estación y guarda los resultados.
    """
    logger.info("Generando reporte de rendimiento espacial (por cuenca y estación)...")

    # 1. Preparar los datos: Unir predicciones con metadatos
    # Usamos los índices del fold de prueba para seleccionar las filas correspondientes
    df_predicciones = df_datos_original.loc[last_fold_data['test_indices']].copy()
    
    # Añadimos los valores reales y predichos del fold
    df_predicciones['y_test'] = last_fold_data['y_test'].values
    df_predicciones['predictions_log'] = last_fold_data['predictions_log']

    # 2. Calcular métricas por CUENCA
    resumen_por_cuenca = df_predicciones.groupby('basin').apply(calcular_metricas_grupo).dropna().round(3)
    
    if not resumen_por_cuenca.empty:
        ruta_cuenca = os.path.join(output_path, f"{run_id}_rendimiento_por_cuenca.csv")
        resumen_por_cuenca.to_csv(ruta_cuenca)
        logger.info(f"  - Reporte por cuenca guardado en: {ruta_cuenca}")

    # 3. Calcular métricas por ESTACIÓN
    resumen_por_estacion = df_predicciones.groupby('station_code').apply(calcular_metricas_grupo).dropna().round(3)

    if not resumen_por_estacion.empty:
        ruta_estacion = os.path.join(output_path, f"{run_id}_rendimiento_por_estacion.csv")
        resumen_por_estacion.to_csv(ruta_estacion)
        logger.info(f"  - Reporte por estación guardado en: {ruta_estacion}")