# -*- coding: utf-8 -*-
"""
Módulo para exportación de datos procesados
"""

import os
import pandas as pd

def limpiar_columnas_para_parquet(df):
    """
    Convierte columnas tipo 'object' con valores numéricos a float, excluyendo columnas clave no numéricas.
    """
    columnas_excluir = ['station_code', 'basin', 'geometry', 'crs']
    columnas_obj = [col for col in df.select_dtypes(include='object').columns if col not in columnas_excluir]

    for col in columnas_obj:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            print(f"⚠️ No se pudo convertir columna '{col}' a numérico: {e}")

    return df



def redondear_variables_numericas(df, decimales=4):
    """
    Redondea todas las columnas numéricas del DataFrame a la cantidad de decimales especificada.
    """
    df_numerico = df.select_dtypes(include='number')
    df[df_numerico.columns] = df_numerico.round(decimales)
    return df


def exportar_para_notebook(df, output_folder, filename="datos_completos_notebook.parquet"):
    """
    Exporta el DataFrame completo en formato Parquet para exploración en Jupyter Notebook.

    Args:
        df (pd.DataFrame): DataFrame procesado final con geometría incluida.
        output_folder (str): Ruta donde guardar el archivo.
        filename (str): Nombre del archivo de salida.
    """
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)

    df_export = df.copy()
    df_export = limpiar_columnas_para_parquet(df_export)
    df_export = redondear_variables_numericas(df_export)

    try:
        df_export.to_parquet(output_path, index=False)
        print(f"📤 Archivo exportado para notebook: {output_path}")
    except Exception as e:
        print(f"❌ Error al exportar archivo Parquet: {e}")


def exportar_datos_finales(df, carpeta="data/processed"):
    """
    Exporta el DataFrame completo en formatos Parquet y CSV, con valores redondeados y limpiados.

    Args:
        df (pd.DataFrame): DataFrame final.
        carpeta (str): Carpeta de destino.
    """
    os.makedirs(carpeta, exist_ok=True)

    df_parquet = df.copy()
    df_parquet = limpiar_columnas_para_parquet(df_parquet)
    df_parquet = redondear_variables_numericas(df_parquet)

    df_csv = df.copy()
    df_csv = redondear_variables_numericas(df_csv)

    csv_path = os.path.join(carpeta, "datos_completos.csv")
    parquet_path = os.path.join(carpeta, "datos_completos.parquet")

    try:
        df_parquet.to_parquet(parquet_path, index=False)
        df_csv.to_csv(csv_path, index=False)
        print("💾 Datos finales exportados en formato Parquet y CSV.")
    except Exception as e:
        print(f"❌ Error al exportar datos finales: {e}")
        
        
        
def test_exportaciones(carpeta="data/processed"):
    """
    Verifica que los archivos exportados existan, no estén vacíos y contengan las columnas esperadas.
    """
    import pyarrow.parquet as pq

    csv_path = os.path.join(carpeta, "datos_completos.csv")
    parquet_path = os.path.join(carpeta, "datos_completos.parquet")

    errores = []

    # 1. Verificar existencia
    if not os.path.isfile(csv_path):
        errores.append("❌ No se encontró el archivo CSV exportado.")
    if not os.path.isfile(parquet_path):
        errores.append("❌ No se encontró el archivo Parquet exportado.")

    # 2. Verificar contenido mínimo
    if os.path.isfile(csv_path):
        df_csv = pd.read_csv(csv_path)
        if df_csv.empty:
            errores.append("⚠️ El archivo CSV está vacío.")
        else:
            columnas_necesarias = {'station_code', 'basin'}
            if not columnas_necesarias.issubset(df_csv.columns):
                errores.append("❌ El archivo CSV no contiene las columnas necesarias.")

    if os.path.isfile(parquet_path):
        try:
            df_parquet = pd.read_parquet(parquet_path)
            if df_parquet.empty:
                errores.append("⚠️ El archivo Parquet está vacío.")
            else:
                columnas_necesarias = {'station_code', 'basin'}
                if not columnas_necesarias.issubset(df_parquet.columns):
                    errores.append("❌ El archivo Parquet no contiene las columnas necesarias.")

                # 3. Verificar redondeo a 4 decimales
                numeric_cols = df_parquet.select_dtypes(include='number')
                for col in numeric_cols.columns:
                    if (numeric_cols[col] % 0.0001 > 1e-8).any():
                        errores.append(f"⚠️ La columna '{col}' tiene más de 4 decimales en Parquet.")
                        break

        except Exception as e:
            errores.append(f"❌ Error al leer archivo Parquet: {e}")

    # Reporte final
    if errores:
        print("🧪 Resultados del test de exportación:")
        for err in errores:
            print(err)
    else:
        print("✅ Test de exportación: todos los archivos están correctos.")

