# =============================================================================
# MAIN - Análisis y Simulación de Datos Hidrobiológicos
# =============================================================================

import os
import pandas as pd
import argparse # <-- Novedad: para control desde la terminal

# Configuración y carga de datos
from config.paths import PATHS
from config.settings import ANALYSIS_PARAMS, ML_PARAMS
from data_processing.data_loader import cargar_datos
from data_processing.basins import integrar_parametros_cuenca, generar_base_cuencas
from data_processing.statistics import resumen_estadistico_por_estacion, frecuencia_muestreo_por_estacion
from data_processing.export import exportar_para_notebook, exportar_datos_finales

# Pipeline de Machine Learning
from ml.pipeline import prepare_data_for_ml, impute_and_scale, train_and_evaluate_models
# Visualización de ML
from visualization.plots import plot_feature_importance, plot_predictions_vs_actual

# Modelos y herramientas de sklearn
from sklearn.model_selection import train_test_split


def ejecutar_analisis_exploratorio(df):
    """
    Encapsula toda la lógica del análisis exploratorio y la exportación de resúmenes.
    """
    print("\n--- INICIANDO PARTE A: ANÁLISIS EXPLORATORIO ---")
    
    print("📈 Generando reportes estadísticos...")
    resumen = resumen_estadistico_por_estacion(df, ANALYSIS_PARAMS['variables_principales'])
    frecuencia = frecuencia_muestreo_por_estacion(df)

    resumen.to_excel("salidas/resumenes_estadisticos/por_estacion.xlsx")
    frecuencia.to_excel("salidas/resumenes_estadisticos/frecuencia_muestreo.xlsx")
    print("✅ Resúmenes estadísticos exportados correctamente.")

    print("💾 Exportando archivos de datos procesados...")
    exportar_para_notebook(df, output_folder="data/processed")
    exportar_datos_finales(df)
    print("✅ Archivos de datos exportados correctamente.")


def ejecutar_pipeline_ml(df):
    """
    Encapsula el flujo de trabajo completo de Machine Learning.
    """
    print("\n--- INICIANDO PARTE B: SIMULACIÓN CON MACHINE LEARNING ---")

    # 1. Preparar datos
    X, y = prepare_data_for_ml(df, ML_PARAMS['TARGET'], ML_PARAMS['FEATURES'])
    
    # 2. División de datos (temporal)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ML_PARAMS['TEST_SIZE'], shuffle=False
    )

    # 3. Imputar y Escalar
    X_train, X_test, X_train_scaled, X_test_scaled = impute_and_scale(X_train, X_test)

    # 4. Entrenar y Evaluar Modelos
    rf_model, results = train_and_evaluate_models(
        X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled,
        ML_PARAMS['RANDOM_STATE'], ML_PARAMS['RF_N_ESTIMATORS'], PATHS['output_models']
    )
    
    print("\n--- Resultados de la Evaluación ---")
    results_df = pd.DataFrame(results).round(3)
    print(results_df)
    results_df.to_csv("salidas/resumenes_estadisticos/ml_results.csv")


    # 5. Visualizar Resultados del Mejor Modelo (Random Forest)
    print("\n🎨 Generando gráficos de resultados de ML...")
    rf_preds = rf_model.predict(X_test)
    plot_feature_importance(rf_model, ML_PARAMS['FEATURES'], PATHS['output_ml_plots'])
    plot_predictions_vs_actual(y_test, rf_preds, "Random Forest", PATHS['output_ml_plots'])


def main(args):
    """
    Función principal que orquesta la ejecución del proyecto.
    """
    # --- Carga y preparación inicial de datos (común a ambos flujos) ---
    print("📥 Cargando y preparando datos base...")
    generar_base_cuencas(
        data_folder=r"G:\Mi unidad\Formación Facultad\01_Maestría\02_Laboratorio de Mecanica de los Fluidos\02_data\_base\Shapes\_cuencas",
        output_folder="data/processed"
    )
    df_inicial = cargar_datos(PATHS['input_data'])
    df_completo = integrar_parametros_cuenca(df_inicial, "data/processed/cuencas_info_wide.csv")
    print("✅ Datos cargados e integrados.")

    # --- Ejecución selectiva de flujos de trabajo ---
    if args.run_eda or args.run_all:
        ejecutar_analisis_exploratorio(df_completo)
        
    if args.run_ml or args.run_all:
        ejecutar_pipeline_ml(df_completo)

    print("\n🎉 Proceso completado exitosamente.")


if __name__ == "__main__":
    # Novedad: Añadimos un parser de argumentos para controlar la ejecución
    parser = argparse.ArgumentParser(description="Pipeline de análisis y simulación de datos hidrobiológicos.")
    
    parser.add_argument('--run-eda', action='store_true', help="Ejecuta solamente el análisis exploratorio de datos.")
    parser.add_argument('--run-ml', action='store_true', help="Ejecuta solamente el pipeline de Machine Learning.")
    # Si no se especifica ninguna bandera, asumimos que se quiere correr todo
    args = parser.parse_args()
    
    # Si no se pasó ningún argumento específico, activamos 'run_all'
    if not (args.run_eda or args.run_ml):
        args.run_all = True
    else:
        args.run_all = False

    main(args)