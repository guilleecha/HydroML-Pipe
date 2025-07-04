"""
Módulo para operaciones relacionadas con cuencas: mapeo, validación, integración morfométrica
"""

import pandas as pd
import os
import geopandas as gpd

CUENCAS_HEREDADAS = {
    'AMO0': 'MO1',
    'AMO1': 'MO1',
    'AMO2': 'MO1',
    'LR1':  'MO1',
    'LR2':  'MO1',
    'LR3':  'MO1',
}

def mapear_cuencas(station_name):
    """Asigna cada estación a su cuenca correspondiente basado en su código"""
    clean_name = str(station_name).strip().upper()
    rules = {
        'MO': 'Arroyo Molino',
        'AMO': 'Arroyo Molino',
        'LR': 'Arroyo Molino',
        'P': 'Arroyo Pantanoso',
        'LP': 'Arroyo Las Piedras',
        'M': 'Arroyo Miguelete',
        'MN': 'Arroyo Carrasco',
        'TO': 'Arroyo Carrasco',
        'CDCH': 'Arroyo Carrasco',
        'CDCN': 'Arroyo Carrasco',
        'CA': 'Arroyo Carrasco',
        '99': 'Sin Clasificar',
        '999': 'Sin Clasificar'
    }
    for prefix, basin in rules.items():
        if clean_name.startswith(prefix):
            return basin
    return 'Otras Cuencas'

def validar_asignaciones(df):
    """
    Valida la asignación de cuencas y geometrías en el DataFrame geo-referenciado.
    Retorna un string con un reporte resumen.
    """
    reporte = []

    total = len(df)
    reporte.append(f"🔎 Total de registros: {total}")

    # Validar columna basin
    if 'basin' not in df.columns:
        reporte.append("❌ No se encontró la columna 'basin'")
    else:
        sin_cuenca = df['basin'].isna().sum()
        reporte.append(f"❗ Estaciones sin cuenca asignada: {sin_cuenca}")
        resumen = df['basin'].value_counts().to_dict()
        reporte.append("✅ Distribución por cuenca:")
        for nombre, count in resumen.items():
            reporte.append(f"   - {nombre}: {count} muestras")

    # Validar geometría
    if 'geometry' not in df.columns:
        reporte.append("❌ No se encontró la columna 'geometry'")
    else:
        sin_geom = df['geometry'].isna().sum()
        reporte.append(f"❗ Estaciones sin geometría asignada: {sin_geom}")

    # Validar duplicados
    if 'station_code' in df.columns and 'fecha' in df.columns:
        duplicados = df.duplicated(subset=['station_code', 'fecha']).sum()
        reporte.append(f"🔁 Registros duplicados (station_code + fecha): {duplicados}")

    return "\n".join(reporte)




def integrar_parametros_cuenca(df, path_cuencas_csv, how="left"):
    """
    Une los parámetros morfométricos de cuenca al DataFrame principal según 'station_code'.

    Args:
        df (pd.DataFrame): DataFrame principal con columna 'station_code'.
        path_cuencas_csv (str): Ruta al archivo cuencas_info_wide.csv.
        how (str): Tipo de merge ('left', 'inner', etc.).

    Returns:
        pd.DataFrame: DataFrame enriquecido.
    """
    print("🧩 Integrando parámetros morfométricos de cuenca...")
    df_cuencas = pd.read_csv(path_cuencas_csv)

    # Estandarizar claves
    df['station_code'] = df['station_code'].str.strip().str.upper()
    df_cuencas['station_code'] = df_cuencas['station_code'].str.strip().str.upper()

    # Agregar herencias desde MO1
    for hija, madre in CUENCAS_HEREDADAS.items():
        if hija not in df_cuencas['station_code'].values:
            base_row = df_cuencas[df_cuencas['station_code'] == madre.upper()]
            if not base_row.empty:
                nueva = base_row.copy()
                nueva['station_code'] = hija  # cambiar código heredado
                df_cuencas = pd.concat([df_cuencas, nueva], ignore_index=True)
                print(f"🔁 Heredando datos de {madre} para {hija}")
            else:
                print(f"⚠️ No se encontró información base para heredar a {hija} desde {madre}")

    # Merge con todas las estaciones ahora presentes
    df_merge = df.merge(df_cuencas, on='station_code', how=how, suffixes=('', '_cuenca'))

    # Diagnóstico de faltantes
    sin_match = df_merge[df_merge['basin'].isna()]['station_code'].unique()
    if len(sin_match) > 0:
        print("⚠️ Estaciones sin match en cuencas_info_wide.csv:")
        for est in sin_match:
            print(f"   - {est}")
    else:
        print("✅ Todas las estaciones tienen parámetros de cuenca.")

    return df_merge




def generar_base_cuencas(data_folder, output_folder="data/processed"):
    """
    Recorre archivos shapefile report_*.shp en subdirectorios, genera y guarda las bases
    cuencas_info_wide.csv y cuencas_info_long.csv en formato ancho y largo.

    Args:
        data_folder (str): Carpeta raíz que contiene subcarpetas con archivos report_*.shp
        output_folder (str): Carpeta donde guardar los CSV resultantes
    """

    import traceback

    wide_data = []
    long_data = []

    print("📦 Generando base de datos morfométrica desde archivos .shp...")

    if not os.path.isdir(data_folder):
        raise FileNotFoundError(f"❌ La carpeta especificada no existe:\n{data_folder}")

    print("📁 Explorando carpetas...")
    total_shapefiles = 0

    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.startswith("report_") and file.endswith(".shp"):
                total_shapefiles += 1
                print("🔹 Encontrado:", os.path.join(root, file))

                shapefile_path = os.path.join(root, file)
                try:
                    gdf = gpd.read_file(shapefile_path)
                    print(f"✅ Shapefile leído correctamente: {file} - {len(gdf)} registros")

                    station_code = file.replace("report_", "").replace(".shp", "")
                    basin = mapear_cuencas(station_code)
                    crs = gdf.crs.to_string() if gdf.crs else "Unknown"
                    wkt_geom = gdf.geometry.iloc[0].wkt if not gdf.empty else None

                    gdf_long = gdf.copy()
                    gdf_long["station_code"] = station_code
                    gdf_long["basin"] = basin
                    gdf_long["crs"] = crs
                    long_data.append(gdf_long)

                    wide_row = {
                        "station_code": station_code,
                        "basin": basin,
                        "geometry": wkt_geom,
                        "crs": crs
                    }
                    for _, row in gdf.iterrows():
                        wide_row[row["Parameter"]] = row["Value"]
                    wide_data.append(wide_row)

                except Exception as e:
                    print(f"⚠️ Error leyendo {file}: {e}")
                    traceback.print_exc()

    df_wide = pd.DataFrame(wide_data)
    print(f"🧾 Resumen - Estaciones procesadas: {len(wide_data)}, parámetros en largo: {len(long_data)}")

    if not long_data:
        print("❌ No se encontraron archivos válidos para generar cuencas.")
        return  # ← 🔧 ESTE return sí está bien acá

    df_long = pd.concat(long_data, ignore_index=True)

    os.makedirs(output_folder, exist_ok=True)
    print(f"📁 Carpeta de salida creada o ya existente: {output_folder}")

    csv_wide_path = os.path.join(output_folder, "cuencas_info_wide.csv")
    csv_long_path = os.path.join(output_folder, "cuencas_info_long.csv")

    print(f"💾 Guardando archivos:\n  - {csv_wide_path}\n  - {csv_long_path}")

    df_wide.to_csv(csv_wide_path, index=False)
    df_long.to_csv(csv_long_path, index=False)

    print(f"🧾 Registros generados - WIDE: {len(df_wide)}, LONG: {len(df_long)}")
    print("✅ Bases de cuenca actualizadas correctamente.")

    
    
def test_estaciones_cuencas(expected_estations, path_csv="data/processed/cuencas_info_wide.csv"):
    df = pd.read_csv(path_csv)
    found = set(df["station_code"].str.upper())
    expected = set([e.upper() for e in expected_estations])

    faltantes = expected - found
    if faltantes:
        print("❌ Estaciones faltantes en cuencas_info_wide.csv:")
        for est in faltantes:
            print(f"   - {est}")
    else:
        print("✅ Todas las estaciones esperadas están presentes.")
        
        

