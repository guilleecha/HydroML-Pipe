import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Ruta principal con subcarpetas por cuenca
data_folder = r"G:\Mi unidad\Formación Facultad\01_Maestría\02_Laboratorio de Mecanica de los Fluidos\02_data\_base\Shapes\_cuencas"
os.chdir("G:\\Mi unidad\\Formación Facultad\\01_Maestría\\02_Laboratorio de Mecanica de los Fluidos\\01_scripts")

from data_processing.basins import mapear_cuencas

# Inicializar listas para formato ancho y largo
wide_data = []
long_data = []

# Recorrer todos los archivos en subdirectorios
for root, _, files in os.walk(data_folder):
    for file in files:
        if file.startswith("report_") and file.endswith(".shp"):
            shapefile_path = os.path.join(root, file)

            try:
                gdf = gpd.read_file(shapefile_path)

                # Extraer código de estación y CRS
                station_code = file.replace("report_", "").replace(".shp", "")
                basin = mapear_cuencas(station_code)
                crs = gdf.crs.to_string() if gdf.crs else "Unknown"

                # Geometría como texto WKT (usamos la del primer registro, que es único)
                wkt_geom = gdf.geometry.iloc[0].wkt if not gdf.empty else None

                # Formato largo
                gdf_long = gdf.copy()
                gdf_long["station_code"] = station_code
                gdf_long["basin"] = basin
                gdf_long["crs"] = crs
                long_data.append(gdf_long)

                # Formato ancho
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
                print(f"Error leyendo {file}: {e}")

# Crear DataFrames
df_wide = pd.DataFrame(wide_data)
df_long = pd.concat(long_data, ignore_index=True)

# Guardar archivos CSV
df_wide.to_csv("cuencas_info_wide.csv", index=False)
df_long.to_csv("cuencas_info_long.csv", index=False)

print("✅ Archivos generados:")
print("- cuencas_info_wide.csv")
print("- cuencas_info_long.csv")
