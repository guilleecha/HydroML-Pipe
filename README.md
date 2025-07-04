# HydroML-Pipe: Hydrological Machine Learning Pipeline

Una plataforma de software en Python diseñada para la experimentación reproducible y el modelado predictivo de la calidad del agua en cuencas hidrográficas.

## Sobre el Proyecto

El objetivo de este proyecto es proporcionar un pipeline de Machine Learning robusto y modular para predecir concentraciones de contaminantes (ej. coliformes fecales) a partir de datos hidro-ambientales. La plataforma permite configurar y ejecutar sistemáticamente múltiples experimentos, facilitando la comparación de diferentes modelos, preprocesamientos y conjuntos de variables.

Este trabajo fue desarrollado como parte del curso [Nombre del Curso] de la [Nombre de la Maestría] en la Universidad de la República.

## Estructura del Repositorio

- **/src**: Contiene todo el código fuente del pipeline (procesamiento, entrenamiento, ploteo, etc.).
- **/scripts**: Contiene scripts de utilidad para análisis complementarios (EDA, sensibilidad de filtros) y para la configuración inicial.
- **/data**: Carpeta para almacenar los datos. La subcarpeta `/raw` (ignorada por Git) debe ser creada por el usuario para colocar los datos de entrada.
- `requirements.txt`: Lista de las dependencias de Python necesarias.
- `main.py`: Punto de entrada principal para ejecutar los experimentos.

## Guía de Inicio Rápido

### Prerrequisitos

- Python 3.9+
- pip

### Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/guilleeecha/HydroML-Pipe.git](https://github.com/guilleeecha/HydroML-Pipe.git)
    cd HydroML-Pipe
    ```

2.  **Crear y activar un entorno virtual (recomendado):**
    ```bash
    python -m venv env
    # En Windows:
    env\Scripts\activate
    # En macOS/Linux:
    source env/bin/activate
    ```

3.  **Instalar las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

### Uso

1.  **Colocar los Datos Crudos:**
    * Crea la carpeta `data/raw/`.
    * Coloca tus archivos de datos (`data_wq.xlsx`, `pluv_INUMET.txt`, etc.) dentro de esta carpeta.

2.  **Generar el Archivo de Configuración (Opcional):**
    * Si es la primera vez o usas nuevos datos, ejecuta el script de setup para generar un `config.py` base.
    ```bash
    python scripts/setup_config.py
    ```

3.  **Definir los Experimentos:**
    * Abre el archivo `src/config.py`.
    * Edita la lista `EXPERIMENTS_TO_RUN` para definir los modelos y configuraciones que deseas probar.

4.  **Ejecutar el Pipeline Principal:**
    * Corre el script principal para ejecutar todos los experimentos.
    ```bash
    python main.py
    ```
    * Todos los resultados, modelos y gráficos se guardarán en la carpeta `outputs/`.

## Contacto

Guillermo Echavarría - **[tu.email@ejemplo.com]**

Link del Proyecto: **[https://github.com/tu_usuario/HydroML-Pipe]**
=======
# HydroML-Pipe
>>>>>>> a25c8f1aed28c22c44b0e068c256bd9dc65f5351
