# src/data_processing.py
# Modulo para el pipeline completo de preprocesamiento de datos.

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from src import config

# Obtenemos la instancia del logger que ya fue configurado en main.py
logger = logging.getLogger('PipelineLogger')

# =============================================================================
# SECCION 1: FUNCIONES DE AYUDA INTERNAS ("PRIVADAS")
# Estas funciones realizan tareas especificas y son llamadas por el orquestador.
# =============================================================================

def _limpiar_nombres_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza los nombres de las columnas de un DataFrame.
    Convierte a minusculas y reemplaza espacios y caracteres especiales por '_'.
    """
    return df.rename(columns=lambda x: (
        str(x).lower().replace(' ', '_').replace('°', '')
        .replace('(', '').replace(')', '').replace('/', '_').strip('_')
    ))


def _convertir_fechas(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Convierte una columna de fecha a formato datetime, manejando tanto
    fechas numericas de Excel como strings.
    """
    if date_col not in df.columns:
        logger.warning(f"ADVERTENCIA: No se encontro la columna de fecha '{date_col}'.")
        return df
    
    if pd.api.types.is_numeric_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(
            df[date_col].apply(lambda x: datetime(1899, 12, 30) + timedelta(days=x) if pd.notnull(x) else pd.NaT)
        )
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    return df




def _procesar_datos_lluvia(path_pluv: str) -> pd.DataFrame:
    """
    Carga y procesa el archivo de lluvia para generar variables de ingenieria de tiempo.
    """
    logger.info("Procesando datos de lluvia para ingenieria de variables...")
    try:
        df_pluv = pd.read_csv(path_pluv, sep='\t', encoding='latin-1')
    except FileNotFoundError:
        logger.error(f"ARCHIVO DE LLUVIA NO ENCONTRADO: {path_pluv}")
        raise

    df_pluv = df_pluv.pipe(_limpiar_nombres_columnas).pipe(_convertir_fechas, date_col=config.DATE_COLUMN)
    df_pluv = df_pluv.set_index(config.DATE_COLUMN).sort_index()

    station_cols = [col for col in df_pluv.columns if col != config.DATE_COLUMN]
    for col in station_cols:
        df_pluv[col] = pd.to_numeric(df_pluv[col], errors='coerce')
        
    df_pluv['precipitacion_promedio_mm'] = df_pluv[station_cols].mean(axis=1)
    
    df_pluv['prec_acum_3d'] = df_pluv['precipitacion_promedio_mm'].rolling(window='3D').sum()
    df_pluv['prec_acum_7d'] = df_pluv['precipitacion_promedio_mm'].rolling(window='7D').sum()
    df_pluv['prec_media_7d'] = df_pluv['precipitacion_promedio_mm'].rolling(window='7D').mean()
    
    logger.info("Nuevas variables de lluvia creadas: 'prec_acum_3d', 'prec_acum_7d', 'prec_media_7d'")
    
    return df_pluv[config.RAIN_FEATURES].reset_index()


def _forzar_columnas_numericas(df: pd.DataFrame, columnas_a_forzar: list) -> pd.DataFrame:
    """
    Intenta convertir una lista de columnas a tipo numerico.
    Valores no convertibles se transformaran en NaN.
    """
    logger.info("Forzando conversion a tipo numerico en columnas de variables...")
    for col in columnas_a_forzar:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _reemplazar_outliers(df: pd.DataFrame, columnas_a_evaluar: list, params: dict) -> pd.DataFrame:
    """
    Detecta y trata outliers usando el método y la estrategia especificados.
    Estrategias: 'remove' (reemplaza por NaN) o 'winsorize' (aplana al límite).
    """
    df_limpio = df.copy()
    method = params.get('method', 'iqr')
    # Nuevo: Lee la estrategia de tratamiento, por defecto es remover.
    strategy = params.get('treatment_strategy', 'remove')
    
    logger.info(f"Tratando outliers con método: {method.upper()} y estrategia: {strategy.upper()}...")
    
    for col in columnas_a_evaluar:
        if col not in df_limpio.columns or df_limpio[col].dropna().empty:
            continue
            
        data_to_check = np.log1p(df_limpio[col].dropna()) if col == config.TARGET_COLUMN else df_limpio[col].dropna()
        
        limite_inferior, limite_superior = None, None

        if method == 'iqr':
            iqr_multiplier = params.get('iqr_multiplier', 1.5)
            Q1, Q3 = data_to_check.quantile(0.25), data_to_check.quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior, limite_superior = Q1 - iqr_multiplier * IQR, Q3 + iqr_multiplier * IQR
        
        elif method == 'mad':
            threshold = params.get('threshold', 3.5)
            median = data_to_check.median()
            mad = (data_to_check - median).abs().median()
            if mad == 0: continue
            z_score_modificado = 0.6745 * (data_to_check - median) / mad
            limite_inferior = data_to_check[z_score_modificado >= -threshold].min()
            limite_superior = data_to_check[z_score_modificado <= threshold].max()
        else:
            logger.warning(f"Método de outlier '{method}' no reconocido. Omitiendo columna '{col}'.")
            continue
            
        if col == config.TARGET_COLUMN:
            limite_inferior, limite_superior = np.expm1(limite_inferior), np.expm1(limite_superior)
            
        # --- LÓGICA DE TRATAMIENTO MODIFICADA ---
        outliers_superiores = df_limpio[col] > limite_superior
        outliers_inferiores = df_limpio[col] < limite_inferior
        
        num_outliers = outliers_superiores.sum() + outliers_inferiores.sum()

        if num_outliers > 0:
            if strategy == 'winsorize':
                logger.info(f"  - Se detectaron {num_outliers} outliers en '{col}'. Serán APLANADOS (Winsorized).")
                df_limpio.loc[outliers_superiores, col] = limite_superior
                df_limpio.loc[outliers_inferiores, col] = limite_inferior
            
            elif strategy == 'remove':
                logger.info(f"  - Se detectaron {num_outliers} outliers en '{col}'. Serán REEMPLAZADOS por NaN.")
                df_limpio.loc[outliers_superiores | outliers_inferiores, col] = np.nan
            
            else:
                 logger.warning(f"Estrategia '{strategy}' no reconocida. No se aplicará tratamiento en '{col}'.")

    return df_limpio


def _integrar_parametros_cuenca(df: pd.DataFrame, path_cuencas_csv: str, cuencas_heredadas: dict) -> pd.DataFrame:
    """Une los parametros morfometricos de cuenca al DataFrame principal."""
    logger.info("Integrando parametros morfometricos de cuenca...")
    try:
        df_cuencas = pd.read_csv(path_cuencas_csv)
    except FileNotFoundError:
        logger.error(f"ERROR: No se encontro el archivo de cuencas en {path_cuencas_csv}.")
        raise

    df['station_code'] = df['station_code'].str.strip().str.upper()
    df_cuencas['station_code'] = df_cuencas['station_code'].str.strip().str.upper()

    for hija, madre in cuencas_heredadas.items():
        if hija not in df_cuencas['station_code'].values:
            base_row = df_cuencas[df_cuencas['station_code'] == madre.upper()]
            if not base_row.empty:
                nueva = base_row.copy(); nueva['station_code'] = hija
                df_cuencas = pd.concat([df_cuencas, nueva], ignore_index=True)
    
    return df.merge(df_cuencas, on='station_code', how="left", suffixes=('', '_cuenca'))


def aplicar_filtro_robusto(df: pd.DataFrame, exp_config: dict) -> tuple:
    """
    Aplica el filtro de 3 etapas usando los parametros del experimento especifico.
    La Etapa 3 ahora se enfoca solo en la completitud de las variables de calidad de agua.
    """
    logger.info("Aplicando filtro de 3 etapas...")
    params = exp_config['filter_params']
    feature_set = exp_config['feature_set']
    target_col = config.TARGET_COLUMN
    
    # Etapa 1: Filtrar estaciones con pocas muestras del target
    estaciones_iniciales = set(df['station_code'].unique())
    target_counts = df[df[target_col].notna()]['station_code'].value_counts()
    estaciones_preseleccionadas = set(target_counts[target_counts >= params['min_samples_target']].index.tolist())
    logger.info(f"  [Etapa 1] {len(estaciones_iniciales)} estaciones totales -> {len(estaciones_preseleccionadas)} estaciones preseleccionadas.")
    estaciones_descartadas_e1 = estaciones_iniciales - estaciones_preseleccionadas
    if estaciones_descartadas_e1:
        logger.info(f"    - Descarte Etapa 1: {len(estaciones_descartadas_e1)} estaciones por no tener >= {params['min_samples_target']} muestras del target.")
        logger.info(f"      Estaciones eliminadas: {sorted(list(estaciones_descartadas_e1))}")
    if not estaciones_preseleccionadas:
        logger.warning("ADVERTENCIA: Ninguna estacion paso el filtro de la Etapa 1.")
        return pd.DataFrame(), [], []
    df_preseleccionado = df[df['station_code'].isin(estaciones_preseleccionadas)]

    # Etapa 2: Filtrar variables de calidad de agua por completitud global
    df_con_target = df_preseleccionado[df_preseleccionado[target_col].notna()]
    completeness_global = df_con_target.notna().mean() * 100
    wq_features_candidatas = [var for var in config.WATER_QUALITY_FEATURES if var in df.columns]
    wq_features_filtradas = [var for var in wq_features_candidatas if completeness_global.get(var, 0) >= params['completeness_global']]
    logger.info(f"  [Etapa 2] Se evaluaron {len(wq_features_candidatas)} variables de calidad de agua -> {len(wq_features_filtradas)} pasaron el filtro.")
    wq_features_descartadas = set(wq_features_candidatas) - set(wq_features_filtradas)
    if wq_features_descartadas:
        logger.info(f"    - Descarte Etapa 2: {len(wq_features_descartadas)} variables por tener < {params['completeness_global']}% de completitud global.")
        logger.info(f"      Variables eliminadas: {sorted(list(wq_features_descartadas))}")

    features_clave = wq_features_filtradas + config.RAIN_FEATURES + config.BASIN_FEATURES
    features_clave = [f for f in features_clave if f in feature_set and f in df.columns]

    # Etapa 3: Filtrar estaciones por completitud de las variables de CALIDAD DE AGUA
    if not wq_features_filtradas:
        logger.warning("ADVERTENCIA: No hay suficientes variables de calidad de agua para evaluar la robustez de las estaciones en la Etapa 3.")
        estaciones_finales = estaciones_preseleccionadas
    else:
        k_min = params['k_min_features']
        logger.info(f"  [Etapa 3] Evaluando estaciones. Criterio: deben tener al menos {k_min} de las {len(wq_features_filtradas)} variables de calidad de agua clave.")
        completeness_local = df_preseleccionado.groupby('station_code')[wq_features_filtradas].apply(lambda x: x.notna().mean() * 100)
        estaciones_robustas_check = (completeness_local >= params['completeness_station']).sum(axis=1)
        estaciones_finales = set(estaciones_robustas_check[estaciones_robustas_check >= k_min].index.tolist())
    
    logger.info(f"  - Resultado: {len(estaciones_preseleccionadas)} preseleccionadas -> {len(estaciones_finales)} finales.")
    estaciones_descartadas_e3 = estaciones_preseleccionadas - estaciones_finales
    if estaciones_descartadas_e3:
        logger.info(f"    - Descarte Etapa 3: {len(estaciones_descartadas_e3)} estaciones por no cumplir el criterio de robustez.")
        logger.info(f"      Estaciones eliminadas: {sorted(list(estaciones_descartadas_e3))}")

    if not estaciones_finales:
        logger.warning("ADVERTENCIA: Ninguna estacion paso el filtro final de la Etapa 3.")
        return pd.DataFrame(), [], []

    # Creacion del DataFrame Final
    final_columns = config.METADATA_COLUMNS + [target_col] + features_clave
    df_final = df[df['station_code'].isin(estaciones_finales)][final_columns].copy()
    logger.info(f"OK: Filtro de 3 etapas completado. Dataset final con {df_final.shape[0]} filas y {len(features_clave)} features.")
    
    return df_final, sorted(list(estaciones_finales)), sorted(features_clave)


# En src/data_processing.py

def filtrar_por_casos_completos(df, exp_config):
    """
    Filtra el DataFrame en dos etapas, con un log detallado de los datos descartados.
    """
    logger.info("Aplicando filtro de 2 etapas...")
    
    feature_set = exp_config['feature_set']
    target_col = config.TARGET_COLUMN
    station_col = 'station_code'
    
    n_filas_inicial = len(df)
    estaciones_iniciales = set(df[station_col].unique())
    logger.info(f"Estado Inicial: {n_filas_inicial} filas y {len(estaciones_iniciales)} estaciones.")

    # --- ETAPA 1: FILTRO POR CASOS COMPLETOS ---
    columnas_a_evaluar = [target_col] + [f for f in feature_set if f in df.columns]
    df_casos_completos = df.dropna(subset=columnas_a_evaluar)
    
    n_filas_intermedias = len(df_casos_completos)
    filas_perdidas_etapa1 = n_filas_inicial - n_filas_intermedias
    logger.info(f"  - Etapa 1 (Casos Completos para {len(columnas_a_evaluar)-1} variables): {n_filas_inicial} -> {n_filas_intermedias} filas. (Se perdieron {filas_perdidas_etapa1} filas por datos NaN)")

    if df_casos_completos.empty:
        logger.warning("No quedaron datos después del filtro de casos completos.")
        return pd.DataFrame(), None, []

    # --- ETAPA 2: FILTRO POR MÍNIMO DE MUESTRAS POR ESTACIÓN ---
    filter_params = exp_config.get('filter_params', {})
    min_muestras = filter_params.get('min_station_samples', 0)

    if min_muestras > 0:
        estaciones_intermedias = set(df_casos_completos[station_col].unique())
        
        counts = df_casos_completos[station_col].value_counts()
        estaciones_a_mantener = set(counts[counts >= min_muestras].index)
        
        df_final = df_casos_completos[df_casos_completos[station_col].isin(estaciones_a_mantener)]
        
        # --- NUEVO: Detalle de estaciones descartadas ---
        estaciones_descartadas = estaciones_intermedias - estaciones_a_mantener
        n_filas_final = len(df_final)
        filas_perdidas_etapa2 = n_filas_intermedias - n_filas_final
        
        logger.info(f"  - Etapa 2 (Mínimo por Estación >= {min_muestras}): {len(estaciones_intermedias)} -> {len(estaciones_a_mantener)} estaciones.")
        if estaciones_descartadas:
            logger.info(f"    - Estaciones descartadas ({len(estaciones_descartadas)}): {sorted(list(estaciones_descartadas))}")
            logger.info(f"    - Esto resultó en la pérdida de {filas_perdidas_etapa2} filas adicionales.")
    else:
        df_final = df_casos_completos
        logger.info("  - Etapa 2 (Mínimo por Estación): Omitida.")

    logger.info(f"Filtrado finalizado. Total de filas conservadas: {len(df_final)} de {n_filas_inicial} ({len(df_final)/n_filas_inicial:.2%})")
    
    features_finales = [f for f in feature_set if f in df_final.columns]
    
    return df_final, None, features_finales


# En src/data_processing.py (puedes ponerla con las otras funciones de ayuda)

def _add_features_estacionales(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Crea dos nuevas columnas con la transformación seno y coseno del día del año
    para capturar la estacionalidad.
    """
    logger.info("Añadiendo características de estacionalidad (seno/coseno)...")
    
    # Asegurarse de que la columna de fecha sea de tipo datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Obtener el día del año (de 1 a 366 para años bisiestos)
    dia_del_año = df[date_col].dt.dayofyear
    dias_en_año = df[date_col].dt.is_leap_year.apply(lambda x: 366 if x else 365)
    
    # Transformación seno y coseno
    df['seasonal_sin'] = np.sin(2 * np.pi * dia_del_año / dias_en_año)
    df['seasonal_cos'] = np.cos(2 * np.pi * dia_del_año / dias_en_año)
    
    return df


def _procesar_datos_temperatura(path_temp: str) -> pd.DataFrame:
    """
    Carga y procesa el archivo de INUMET para generar features de temperatura
    y heliofanía, manejando correctamente los nombres de las columnas.
    """
    logger.info("Procesando datos de temperatura ambiente y heliofanía...")
    try:
        df_temp = pd.read_csv(path_temp, sep='\t', header=[0, 1])
    except FileNotFoundError:
        logger.error(f"ARCHIVO DE TEMPERATURA NO ENCONTRADO: {path_temp}")
        raise

    # Aplanar el MultiIndex de las columnas
    df_temp.columns = ['_'.join(map(str, col)).strip().lower() for col in df_temp.columns.values]
    
    # --- CORRECCIÓN: Encontrar y renombrar la columna de fecha de forma robusta ---
    # Buscamos la primera columna que CONTENGA "fecha"
    try:
        date_col_original = [col for col in df_temp.columns if 'fecha' in col][0]
        df_temp = df_temp.rename(columns={date_col_original: 'date'})
    except IndexError:
        logger.error("No se pudo encontrar una columna de fecha en el archivo de temperatura. Abortando.")
        raise
    # --- FIN DE LA CORRECCIÓN ---

    df_temp = _convertir_fechas(df_temp, date_col='date')

    # Procesar Temperatura
    temp_cols = [col for col in df_temp.columns if 'temperatura media' in col]
    for col in temp_cols:
        df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
    df_temp['temp_ambiente_promedio'] = df_temp[temp_cols].mean(axis=1)
    
    # Procesar Heliofanía
    heliofania_col = [col for col in df_temp.columns if 'heliofania' in col][0]
    df_temp = df_temp.rename(columns={heliofania_col: 'horas_sol'})
    df_temp['horas_sol'] = pd.to_numeric(df_temp['horas_sol'], errors='coerce')

    # Ingeniería de Características de Temperatura
    df_temp_ts = df_temp.set_index('date').sort_index()
    df_temp_ts['temp_media_3d'] = df_temp_ts['temp_ambiente_promedio'].rolling(window='3D').mean()
    df_temp_ts['temp_media_7d'] = df_temp_ts['temp_ambiente_promedio'].rolling(window='7D').mean()
    df_temp_ts['temp_media_14d'] = df_temp_ts['temp_ambiente_promedio'].rolling(window='14D').mean()
    
    logger.info("Nuevas variables de temperatura y heliofanía creadas.")
    
    # Devolvemos el DataFrame con las nuevas columnas
    columnas_a_devolver = ['date'] + config.TEMP_FEATURES + config.SOLAR_FEATURES
    return df_temp_ts.reset_index()[[col for col in columnas_a_devolver if col in df_temp_ts.reset_index().columns]]


# =============================================================================
# SECCION 2: FUNCION ORQUESTADORA PRINCIPAL
# =============================================================================

def crear_dataframe_base() -> pd.DataFrame:
    """
    Ejecuta los pasos de procesamiento comunes (carga, fusion, enriquecimiento)
    para crear un unico DataFrame completo y listo para los experimentos.
    """
    logger.info("--- Creando DataFrame Base (Enriquecido y sin Filtrar) ---")
    
    # 1. Carga y limpieza de datos de calidad de agua
    path_wq = config.PATHS['INPUT_DATA_WQ']
    logger.info(f"Cargando datos de calidad de agua desde: {path_wq}")
    valores_nan = ['N/D', 's/d', 'S/D', '-', ' ', '']
    df_wq = pd.read_excel(
        path_wq, sheet_name='data', 
        engine='openpyxl', na_values=valores_nan
    )
    df_wq = df_wq.pipe(_limpiar_nombres_columnas).pipe(_convertir_fechas, date_col=config.DATE_COLUMN)
    df_wq = _forzar_columnas_numericas(df_wq, config.ALL_PREDICTOR_VARIABLES)
    
    # 2. Procesamiento de datos de lluvia
    df_pluv_procesado = _procesar_datos_lluvia(config.PATHS['INPUT_DATA_PLUV'])
    
    # --- CORRECCIÓN: Se procesan los datos de temperatura aquí ---
    # 3. Procesamiento de datos de temperatura y heliofanía
    df_temp_procesado = _procesar_datos_temperatura(config.PATHS['INPUT_DATA_TEMP'])
    
    # 4. Fusión de todas las fuentes de datos
    logger.info("Fusionando datos de calidad de agua, lluvia y temperatura...")
    df_fusionado = pd.merge(df_wq, df_pluv_procesado, on=config.DATE_COLUMN, how='left')
    
    # --- CORRECCIÓN: Añadida la fusión de los datos de temperatura ---
    df_fusionado = pd.merge(df_fusionado, df_temp_procesado, on=config.DATE_COLUMN, how='left')

    # Rellenar NaNs en features de lluvia (asumimos que si no hay dato es que no llovió)
    for col in config.RAIN_FEATURES:
        if col in df_fusionado.columns: df_fusionado[col] = df_fusionado[col].fillna(0)
    
    # 5. Enriquecimiento con datos de cuenca y estacionalidad
    df_enriquecido = _integrar_parametros_cuenca(df_fusionado, config.PATHS['INPUT_DATA_CUENCAS'], config.CUENCAS_HEREDADAS)
    df_con_estacionalidad = _add_features_estacionales(df_enriquecido, config.DATE_COLUMN)
    
    # 6. Ordenado final
    df_ordenado = df_con_estacionalidad.sort_values(by=config.DATE_COLUMN).reset_index(drop=True)
    
    logger.info(f"OK: DataFrame Base creado y ordenado exitosamente con {df_ordenado.shape[0]} filas.")
    return df_ordenado