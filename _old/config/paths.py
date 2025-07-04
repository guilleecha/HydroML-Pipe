import os

# =============================================================================
# DEFINICIÓN DINÁMICA DE LA RAÍZ DEL PROYECTO
# =============================================================================

# 1. Obtiene la ruta absoluta del archivo actual (paths.py)
#    ej: G:\Mi unidad\...\01_scripts\config\paths.py
current_file_path = os.path.abspath(__file__)

# 2. Obtiene el directorio que contiene a este archivo (la carpeta 'config')
#    ej: G:\Mi unidad\...\01_scripts\config
config_dir = os.path.dirname(current_file_path)

# 3. La raíz de nuestro proyecto ('PROJECT_ROOT') es el directorio padre de 'config'
#    ej: G:\Mi unidad\...\01_scripts
PROJECT_ROOT = os.path.dirname(config_dir)

# ¡Ahora podemos construir todas las rutas de forma segura y portable!

# =============================================================================
# DICCIONARIO DE RUTAS (PATHS)
# =============================================================================

PATHS = {
    # Usamos os.path.join() para que funcione en cualquier sistema operativo (Windows, Linux, Mac)
    
    # Rutas de datos de entrada
    'input_data': os.path.join(PROJECT_ROOT, 'data', 'raw', 'datos_completos.csv'), # Asumo que está en data/raw/
    'processed_data': os.path.join(PROJECT_ROOT, 'data', 'processed'),
    
    # Rutas de salida
    'output_summaries': os.path.join(PROJECT_ROOT, 'salidas', 'resumenes_estadisticos'),
    'output_final_data': os.path.join(PROJECT_ROOT, 'salidas', 'datos_finales'),
    'output_models': os.path.join(PROJECT_ROOT, 'salidas', 'modelos'),
    'output_ml_plots': os.path.join(PROJECT_ROOT, 'salidas', 'graficos', 'ml'),
    'output_eda_plots': os.path.join(PROJECT_ROOT, 'salidas', 'graficos', 'eda')
}

# --- Creación automática de directorios de salida ---
# Esto asegura que las carpetas para guardar resultados siempre existan.
for key, path in PATHS.items():
    if 'output' in key: # Solo crea carpetas de salida
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directorio creado: {path}")