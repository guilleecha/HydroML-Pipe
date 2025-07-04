# scripts/generar_info_cuencas.py
# Script de utilidad para procesar shapefiles y generar los CSV de información de cuencas.

import os
import sys
import pandas as pd
import geopandas as gpd
import traceback

# Añadimos la raíz del proyecto al path de Python para poder importar 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src import config # Importamos nuestra configuracion centralizada

def _limpiar_texto(texto: str) -> str:
    """Función de ayuda que estandariza un string."""
    return (
        str(texto).lower().replace(' ', '_').replace('°', '')
        .replace('(', '').replace(')', '').replace('/', '_').strip('_')
    )

def _limpiar_nombres_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Estandariza los NOMBRES DE COLUMNA de un DataFrame."""
    return df.rename(columns=lambda x: _limpiar_texto(x))

def mapear_cuencas(station_name):
    """Asigna cada estación a su cuenca correspondiente basado en su código."""
    # ... (código de la función sin cambios) ...
    clean_name = str(station_name).strip().upper()
    rules = {
        'MO': 'Arroyo Molino', 'AMO': 'Arroyo Molino', 'LR': 'Arroyo Molino',
        'P': 'Arroyo Pantanoso', 'LP': 'Arroyo Las Piedras', 'M': 'Arroyo Miguelete',
        'MN': 'Arroyo Carrasco', 'TO': 'Arroyo Carrasco', 'CDCH': 'Arroyo Carrasco',
        'CDCN': 'Arroyo Carrasco', 'CA': 'Arroyo Carrasco', '99': 'Sin Clasificar',
        '999': 'Sin Clasificar'
    }
    for prefix, basin in rules.items():
        if clean_name.startswith(prefix):
            return basin
    return 'Otras Cuencas'


def procesar_shapefiles_a_csv():
    """
    Función principal que lee shapefiles, los procesa y guarda los resultados en CSV.
    """
    data_folder = config.PATHS['INPUT_GIS_SHAPEFILES']
    output_folder = os.path.dirname(config.PATHS['INPUT_DATA_CUENCAS'])

    print("[INFO] Generando base de datos morfométrica desde archivos .shp...")
    print(f"[INFO] Buscando en: {data_folder}")

    if not os.path.isdir(data_folder):
        raise FileNotFoundError(f"[ERROR] La carpeta de shapefiles especificada en config.py no existe:\n{data_folder}")

    wide_data = []
    long_data = []

    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.startswith("report_") and file.endswith(".shp"):
                shapefile_path = os.path.join(root, file)
                try:
                    gdf = gpd.read_file(shapefile_path)
                    station_code = file.replace("report_", "").replace(".shp", "")
                    basin = mapear_cuencas(station_code)
                    crs = gdf.crs.to_string() if gdf.crs else "Unknown"
                    wkt_geom = gdf.geometry.iloc[0].wkt if not gdf.empty else None

                    # Formato largo
                    gdf_long = gdf.copy()
                    gdf_long["station_code"] = station_code
                    gdf_long["basin"] = basin
                    gdf_long["crs"] = crs
                    long_data.append(gdf_long)

                    # Formato ancho
                    wide_row = {"station_code": station_code, "basin": basin, "geometry": wkt_geom, "crs": crs}
                    for _, row in gdf.iterrows():
                        wide_row[row["Parameter"]] = row["Value"]
                    wide_data.append(wide_row)
                except Exception as e:
                    print(f"[ERROR] Error leyendo {file}: {e}")
                    traceback.print_exc()

    if not wide_data:
        print("[ERROR] No se procesó ningún shapefile válido. Terminando.")
        return

    # Crear DataFrames
    df_wide_raw = pd.DataFrame(wide_data)
    df_long_raw = pd.concat(long_data, ignore_index=True) if long_data else pd.DataFrame()
    
    # Limpiar el DataFrame WIDE
    print("[INFO] Limpiando nombres de columnas del archivo WIDE...")
    df_wide_clean = _limpiar_nombres_columnas(df_wide_raw)
    
    # Limpiar el DataFrame LONG
    print("[INFO] Limpiando nombres de columnas y valores del archivo LONG...")
    df_long_clean = _limpiar_nombres_columnas(df_long_raw)
    
    # --- LÓGICA CLAVE CORREGIDA ---
    # Revisamos si la columna 'parameter' existe antes de intentar limpiarla
    if 'parameter' in df_long_clean.columns:
        # Aplicamos la limpieza a los VALORES de la columna 'parameter'
        df_long_clean['parameter'] = df_long_clean['parameter'].apply(_limpiar_texto)
    # --- FIN DE LA CORRECCIÓN ---

    # Guardar archivos
    os.makedirs(output_folder, exist_ok=True)
    wide_path = config.PATHS['INPUT_DATA_CUENCAS']
    long_path = os.path.join(output_folder, "cuencas_info_long.csv")
    
    df_wide_clean.to_csv(wide_path, index=False)
    df_long_clean.to_csv(long_path, index=False)
    
    print("\n[INFO] Archivos generados con nombres y valores estandarizados:")
    print(f"  - {wide_path}")
    print(f"  - {long_path}")

if __name__ == '__main__':
    procesar_shapefiles_a_csv()