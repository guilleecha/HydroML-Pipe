# src/clustering.py
# Módulo para realizar el clustering de estaciones basado en sus características.

import os
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Importamos la configuración para saber dónde guardar los resultados cacheados.
from src import config

# Obtenemos la instancia del logger.
logger = logging.getLogger('PipelineLogger')

def get_station_clusters(df_cuencas, k, feature_set):
    """
    Agrupa las estaciones en 'k' clusters usando K-Means basado en sus características.
    Implementa un sistema de caché para evitar recalcular.

    Args:
        df_cuencas (pd.DataFrame): El DataFrame con la información de las cuencas.
        k (int): El número de clusters a crear.
        feature_set (list): La lista de columnas a usar para el clustering.

    Returns:
        pd.DataFrame: Un DataFrame con las columnas ['station_code', 'cluster_id'].
    """
    # --- Lógica de Caching ---
    cache_dir = os.path.join(config.PATHS['OUTPUTS'], 'clusters')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'station_clusters_k{k}.csv')

    if os.path.exists(cache_file):
        logger.info(f"Cargando clusters pre-calculados desde el caché para k={k}...")
        return pd.read_csv(cache_file)
    
    # --- Si no está en caché, se calcula ---
    logger.info(f"No se encontró caché. Calculando nuevos clusters para k={k}...")

    # 1. Preparar los datos de las estaciones
    df_stations = df_cuencas.drop_duplicates(subset=['station_code']).copy()
    df_stations = df_stations.set_index('station_code')
    
    # Asegurarse de que todas las features existan y no tengan NaNs para el clustering
    features_existentes = [f for f in feature_set if f in df_stations.columns]
    df_cluster_data = df_stations[features_existentes].dropna()

    if df_cluster_data.empty:
        logger.error("No hay suficientes datos de cuenca para realizar el clustering.")
        raise ValueError("DataFrame vacío después de eliminar NaNs en las características de la cuenca.")

    station_index = df_cluster_data.index

    # 2. Escalar las características
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_cluster_data)

    # 3. Aplicar K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(features_scaled)

    # 4. Crear el DataFrame de resultados
    df_results = pd.DataFrame({
        'station_code': station_index,
        'cluster_id': cluster_labels
    })

    # 5. Guardar en caché para futuras ejecuciones
    logger.info(f"Guardando los nuevos clusters en caché: {cache_file}")
    df_results.to_csv(cache_file, index=False)

    return df_results