# src/validation.py
# Módulo para funciones de validación y chequeo de calidad de datos.

import pandas as pd
import os

def test_exportaciones(carpeta_salida, nombre_archivo_csv, nombre_archivo_parquet):
    """
    Verifica que los archivos exportados existan, no estén vacíos y contengan columnas clave.
    """
    print("\n🧪 Ejecutando tests de validación sobre los archivos exportados...")
    
    csv_path = os.path.join(carpeta_salida, nombre_archivo_csv)
    parquet_path = os.path.join(carpeta_salida, nombre_archivo_parquet)
    errores = []

    # --- El código interno de tu función test_exportaciones va aquí, sin cambios ---
    # 1. Verificar existencia
    if not os.path.isfile(csv_path):
        errores.append(f"❌ No se encontró el archivo CSV exportado en: {csv_path}")
    if not os.path.isfile(parquet_path):
        errores.append(f"❌ No se encontró el archivo Parquet exportado en: {parquet_path}")

    # ... (resto de la lógica de la función de test) ...

    # Reporte final
    if errores:
        print("--- Resultados del test de exportación ---")
        for err in errores:
            print(f"  - {err}")
    else:
        print("✅ ¡Éxito! Todos los tests de exportación pasaron correctamente.")

    return not errores # Devuelve True si no hay errores, False si los hay