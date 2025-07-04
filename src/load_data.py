# -*- coding: utf-8 -*-
"""
Created on [Fecha]

@author: [Tu nombre]
"""

# =============================================================================
# 1. Importación de bibliotecas
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os

# Configuración de visualización
plt.style.use('ggplot')
pd.options.display.max_columns = 30

# =============================================================================
# 2. Carga de datos
# =============================================================================
# Configurar la ruta del archivo (ajustar según tu sistema)
directorio = r'G:\Mi unidad\Formación Facultad\01_Maestría\02_Laboratorio de Mecanica de los Fluidos\02_data\_procesado'  # Cambiar por tu ruta real
nombre_archivo = 'data_wq.xlsx'
ruta_completa = os.path.join(directorio, nombre_archivo)

# Leer el archivo Excel
df = pd.read_excel(
    ruta_completa,
    sheet_name='data',
    skiprows=0,  # Saltar la primera fila de metadatos
    parse_dates=False
)



# =============================================================================
# 3. Limpieza inicial
# =============================================================================
# Eliminar columnas completamente vacías
df = df.dropna(axis=1, how='all')

# Eliminar filas completamente vacías
df = df.dropna(axis=0, how='all')

# Renombrar columnas para consistencia
df.columns = df.columns.str.lower().str.replace(' ', '_')

# =============================================================================
# 4. Conversión de fechas (Versión mejorada)
# =============================================================================
# Convertir la columna 'date' a datetime
df['fecha'] = pd.to_datetime(
    df['date'].apply(
        lambda x: (datetime(1899, 12, 30) + timedelta(days=x)) if pd.notnull(x) else pd.NaT
    )
)

# Eliminar la columna numérica original
df = df.drop(columns=['date'])

# =============================================================================
# 5. Análisis exploratorio inicial (Versión corregida)
# =============================================================================
# Ver estructura de los datos
print("="*80)
print("Resumen inicial de datos:")
print(f"- Total de registros: {len(df)}")
print(f"- Rango temporal: {df['fecha'].min().date()} a {df['fecha'].max().date()}")
print(f"- Estaciones únicas: {df['station'].nunique()}")
print("- Variables disponibles:", list(df.columns))

# Estadísticas básicas (Separado para numéricas y categóricas)
print("\nEstadísticas descriptivas numéricas:")
print(df.describe(include='number'))

print("\nResumen variables categóricas:")
print(df.describe(include='object'))

# Datos faltantes por columna
missing_data = df.isnull().mean().sort_values(ascending=False) * 100
print("\nDatos faltantes por columna (%):")
print(missing_data.round(2))

# =============================================================================
# 6. Visualizaciones iniciales (Versión mejorada)
# =============================================================================
# Configurar figura
fig, ax = plt.subplots(2, 1, figsize=(12, 10))

# Gráfico 1: Distribución temporal de muestras
ax[0].hist(df['fecha'], bins=50, edgecolor='black')
ax[0].set_title('Distribución Temporal de Muestras')
ax[0].set_ylabel('Número de Muestras')
ax[0].xaxis.set_major_locator(mdates.YearLocator())
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Gráfico 2: Completitud de variables principales
(missing_data[missing_data < 100]
 .head(10)
 .sort_values()
 .plot(kind='barh', ax=ax[1], color='teal'))
ax[1].set_title('Top 10 Variables con Menos Datos Faltantes')
ax[1].set_xlabel('% Datos Presentes')
ax[1].set_xlim(0, 100)

plt.tight_layout()
plt.show()