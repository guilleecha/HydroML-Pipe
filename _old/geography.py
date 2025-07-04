# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 17:22:42 2025

@author: USUARIO
"""
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
from config.settings import ANALYSIS_PARAMS
import pandas as pd
from data_processing.basins import mapear_cuencas



# def integrar_geodata(df: pd.DataFrame, gdf_estaciones: GeoDataFrame) -> GeoDataFrame:
#     """
#     Une los datos hidrobiológicos con los datos geográficos y asigna cuencas.
#     """
#     df = df.copy()
#     gdf_estaciones = gdf_estaciones.copy()

#     # Estandarizar claves para evitar errores en el merge
#     df['station_code'] = df['station_code'].str.strip().str.upper()
#     gdf_estaciones['station_code'] = gdf_estaciones['station_code'].str.strip().str.upper()

#     # Unir los datasets
#     df_geo = df.merge(
#         gdf_estaciones[['station_code', 'Arroyo', 'geometry']],
#         on='station_code',
#         how='left'
#     )

#     # Asignar cuenca
#     df_geo['cuenca'] = df_geo['Arroyo']
#     df_geo['cuenca'] = df_geo['cuenca'].fillna(df_geo['station_code'].apply(mapear_cuencas))

#     # Diagnóstico de unión
#     total = len(df_geo)
#     asignadas = df_geo['geometry'].notna().sum()
#     print(f"📌 Estaciones con geometría: {asignadas} / {total}")

#     sin_geom = df_geo[df_geo['geometry'].isna()]
#     if not sin_geom.empty:
#         print("⚠️ Estaciones sin geometría asignada:")
#         print(sin_geom['station_code'].unique())

#     return GeoDataFrame(df_geo, geometry='geometry', crs=gdf_estaciones.crs)

