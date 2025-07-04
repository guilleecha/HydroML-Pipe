# src/utils.py
# Modulo para funciones de ayuda y utilidades generales del proyecto.

import os
import logging
from src import config

# Obtenemos el logger que ya fue configurado en main.py
logger = logging.getLogger('PipelineLogger')

def setup_directorios():
    """
    Crea los directorios de salida base necesarios si no existen.
    Esta funcion asegura que la carpeta 'outputs' principal este lista.
    """
    logger.info("Verificando y creando directorios base de salida...")
    
    # Lista de claves del config que corresponden a DIRECTORIOS base que queremos crear
    # En la nueva estructura, solo necesitamos asegurar que 'outputs' exista.
    # Las subcarpetas de cada corrida se crean dinamicamente en main.py.
    directorios_a_crear = [
        'OUTPUTS', 
    ]
    
    for key in directorios_a_crear:
        path = config.PATHS.get(key)
        if path:
            os.makedirs(path, exist_ok=True)