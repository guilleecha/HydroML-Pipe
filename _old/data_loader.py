# -*- coding: utf-8 -*-
"""
Módulo para carga y limpieza inicial de datos hidrobiológicos
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import warnings
from config.paths import PATHS
from config.settings import ANALYSIS_PARAMS

def cargar_datos(ruta: Optional[str] = None) -> pd.DataFrame:
    """Carga y realiza limpieza inicial de los datos
    
    Args:
        ruta: Path opcional para sobreescribir la configuración default
    
    Returns:
        pd.DataFrame: Datos limpios y estructurados
    
    Raises:
        FileNotFoundError: Si no encuentra el archivo de entrada
        ValueError: Si el dataframe resultante está vacío
    """
    try:
        file_path = ruta or PATHS['input_data']
        print(f"Cargando datos desde: {file_path}")
        
        # Carga con tipos específicos y manejo de fechas
        df = pd.read_excel(
            file_path,
            sheet_name='data',
            engine='openpyxl',
            parse_dates=False,
            dtype={'station_code': 'string'}
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}") from e

    if df.empty:
        raise ValueError("El archivo está vacío o no contiene datos válidos")

    # Pipeline de limpieza
    df = (df
          .pipe(_eliminar_columnas_vacias)
          .pipe(limpiar_nombres_columnas)
          .pipe(convertir_fechas)
          .pipe(_filtrar_por_umbral_missing)
          .pipe(_validar_estructura_base)
          )
    
    return df

def limpiar_nombres_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Estandariza nombres de columnas
    
    Ejemplo:
        'Temperatura (°C)' -> 'temperatura_c'
    """
    return df.rename(columns=lambda x: (
        str(x).lower()
        .replace(' ', '_')
        .replace('°', '')
        .replace('(', '')
        .replace(')', '')
        .replace('/', '_')
        .strip('_')
    ))

def convertir_fechas(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte diferentes formatos de fecha a datetime
    
    Maneja:
        - Fechas en formato Excel (números seriales)
        - Strings en formato ISO
    """
    if 'date' not in df.columns:
        warnings.warn("No se encontró columna de fechas. Se creará 'fecha' con NaT")
        df['fecha'] = pd.NaT
        return df
    
    # Conversión de fechas Excel
    try:
        if pd.api.types.is_numeric_dtype(df['date']):
            df['fecha'] = pd.to_datetime(
                df['date'].apply(lambda x: datetime(1899, 12, 30) + timedelta(days=x) if pd.notnull(x) else pd.NaT)
            )
        else:
            df['fecha'] = pd.to_datetime(df['date'], errors='coerce')
    except Exception as e:
        raise ValueError("Error en conversión de fechas") from e
    return df.drop(columns=['date'], errors='ignore')

def _eliminar_columnas_vacias(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas y filas completamente vacías"""
    return df.dropna(how='all', axis=1).dropna(how='all', axis=0)

def _filtrar_por_umbral_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra columnas con más del umbral de datos faltantes"""
    umbral = ANALYSIS_PARAMS['umbral_faltantes']
    missing_pct = df.isna().mean()
    
    # Columnas a conservar
    keep_cols = missing_pct[missing_pct <= umbral].index.tolist()
    
    # Conservar siempre columnas clave
    for col in ['station_code', 'fecha']:
        if col in df.columns and col not in keep_cols:
            keep_cols.append(col)
    
    removed_cols = set(df.columns) - set(keep_cols)
    print(f"Columnas eliminadas por faltantes > {umbral*100:.0f}%: {removed_cols}")    
    return df[keep_cols]

def _validar_estructura_base(df: pd.DataFrame) -> pd.DataFrame:
    """Valida la estructura mínima requerida"""
    required_cols = ['station_code', 'fecha']
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Columnas requeridas faltantes: {missing}")
    
    if df['fecha'].isna().all():
        warnings.warn("No hay fechas válidas en el dataset")
    
    return df