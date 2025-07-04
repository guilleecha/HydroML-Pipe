# src/logger_config.py
import logging
import os
from src import config

def setup_logger(log_file_path):
    """
    Configura un logger para escribir en un archivo específico y en la consola.
    Esta función se llama para CADA experimento.
    """
    logger = logging.getLogger('PipelineLogger')
    logger.setLevel(logging.INFO)

    # Limpiar handlers de corridas anteriores para no duplicar mensajes
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # --- LÍNEA AÑADIDA ---
    # Le decimos al logger que no pase los mensajes a su 'padre' (el logger raíz)
    logger.propagate = False
    # --- FIN DE LA LÍNEA AÑADIDA ---

    log_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler para escribir en el archivo de log de la corrida específica
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    # Handler para imprimir en la consola
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)
    logger.addHandler(stream_handler)