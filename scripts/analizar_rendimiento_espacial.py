# scripts/analizar_rendimiento_espacial.py
import pandas as pd
from src.train import nse, pbias # Reutilizamos nuestras funciones
from sklearn.metrics import r2_score

# 1. Cargar los resultados detallados de tu mejor corrida
# Este archivo tiene una fila por cada punto de datos en los folds de prueba
df_detalle = pd.read_csv('outputs/ID_DE_TU_MEJOR_CORRIDA/detailed_fold_results.csv')

# 2. Cargar los datos usados para obtener los metadatos (basin, station_code)
df_datos = pd.read_csv('outputs/ID_DE_TU_MEJOR_CORRIDA/datos_usados_en_modelo.csv')

# Unir ambos para tener predicciones y metadatos juntos
df_completo = pd.merge(df_datos, df_detalle, left_index=True, right_index=True)

# 3. Agrupar por cuenca y calcular las métricas
def calcular_metricas_grupo(grupo):
    return pd.Series({
        'R2': r2_score(grupo['y_test'], grupo['predictions_log']),
        'NSE': nse(grupo['y_test'], grupo['predictions_log']),
        'PBIAS': pbias(grupo['y_test'], grupo['predictions_log'])
    })

resumen_por_cuenca = df_completo.groupby('basin').apply(calcular_metricas_grupo)

print("--- Rendimiento por Cuenca ---")
print(resumen_por_cuenca)