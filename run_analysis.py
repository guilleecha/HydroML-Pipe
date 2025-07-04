# run_analysis.py
# Este script se usa para generar reportes y análisis descriptivos.
# No es parte del pipeline principal del modelo.

import pandas as pd
from src import config
from src.analysis_tools import resumen_estadistico_por_estacion, frecuencia_muestreo_por_estacion

def generar_reportes_descriptivos():
    """
    Carga los datos procesados y genera reportes estadísticos.
    """
    print("--- Iniciando generación de reportes descriptivos ---")
    
    # Decidimos analizar los datos ya procesados y filtrados
    ruta_datos = os.path.join(config.PATHS['processed_data'], 'datos_para_modelo.csv')
    
    try:
        df = pd.read_csv(ruta_datos)
        print(f"Datos cargados exitosamente desde {ruta_datos}")
    except FileNotFoundError:
        print(f"❌ ERROR: No se encontró el archivo de datos procesados.")
        print("    Asegúrate de haber ejecutado 'main.py' al menos una vez.")
        return

    # 1. Generar resumen estadístico
    print("Calculando resumen estadístico por estación...")
    resumen_stats = resumen_estadistico_por_estacion(df, variables=config.ENVIRONMENTAL_VARIABLES)
    
    # 2. Generar análisis de frecuencia
    print("Calculando frecuencia de muestreo por estación...")
    # La función necesita la columna de fecha con el tipo de dato correcto
    df[config.DATE_COLUMN] = pd.to_datetime(df[config.DATE_COLUMN])
    frecuencia_muestreo = frecuencia_muestreo_por_estacion(df)

    # 3. Guardar los resultados en la carpeta de salidas
    ruta_resumen = os.path.join(config.PATHS['outputs'], 'resumen_estadistico.csv')
    ruta_frecuencia = os.path.join(config.PATHS['outputs'], 'resumen_frecuencia_muestreo.csv')

    resumen_stats.to_csv(ruta_resumen)
    frecuencia_muestreo.to_csv(ruta_frecuencia)
    
    print(f"\n✅ Reportes guardados exitosamente en:")
    print(f"  - {ruta_resumen}")
    print(f"  - {ruta_frecuencia}")


if __name__ == '__main__':
    generar_reportes_descriptivos()