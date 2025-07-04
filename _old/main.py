# =============================================================================
# MAIN - Análisis de datos hidrobiológicos
# =============================================================================

# 0. Configuración inicial de entorno
import os

# ⚠️ Establecer ruta de trabajo (solo entorno local)
os.chdir("G:\\Mi unidad\\Formación Facultad\\01_Maestría\\02_Laboratorio de Mecanica de los Fluidos\\01_scripts")


from config.paths import PATHS
from config.settings import ANALYSIS_PARAMS
from data_processing.data_loader import cargar_datos
from data_processing.basins import validar_asignaciones, integrar_parametros_cuenca
from data_processing.statistics import resumen_estadistico_por_estacion, frecuencia_muestreo_por_estacion
from data_processing.export import exportar_para_notebook, exportar_datos_finales
from data_processing.basins import generar_base_cuencas

import subprocess


# --- NUEVOS IMPORTS ---
# Pipeline de Machine Learning
from ml.pipeline import prepare_data_for_ml, impute_and_scale, train_and_evaluate_models
# Visualización de ML
from visualization.plots import plot_feature_importance, plot_predictions_vs_actual



def main():

    # 0. Actualizar base de datos de cuencas (antes de integrar parámetros)
    print(f"📂 Directorio de trabajo actual: {os.getcwd()}")

    generar_base_cuencas(
        data_folder=r"G:\Mi unidad\Formación Facultad\01_Maestría\02_Laboratorio de Mecanica de los Fluidos\02_data\_base\Shapes\_cuencas",
        output_folder="data/processed"
    )

    
    # 1. Cargar y limpiar datos hidrobiológicos
    print("📥 Cargando datos...")
    df = cargar_datos(PATHS['input_data'])

    # 2. Integrar datos espaciales y morfométricos
    print("🌐 Integrando datos de cuenca...")
    df_completo = integrar_parametros_cuenca(df, "data/processed/cuencas_info_wide.csv")

    # 3. Validación básica
    print("✅ Validando asignaciones espaciales...")
    print(validar_asignaciones(df_completo))

    # 4. Análisis estadístico
    print("📈 Generando reportes estadísticos...")
    print("\n🔎 Tipos de datos en variables principales:")
    print(df_completo[ANALYSIS_PARAMS['variables_principales']].dtypes)

    resumen = resumen_estadistico_por_estacion(df_completo, ANALYSIS_PARAMS['variables_principales'])
    frecuencia = frecuencia_muestreo_por_estacion(df_completo)

    resumen.to_excel("salidas/resumenes_estadisticos/por_estacion.xlsx")
    frecuencia.to_excel("salidas/resumenes_estadisticos/frecuencia_muestreo.xlsx")
    print("✅ Resúmenes exportados correctamente.")

    # 5. Exportaciones para uso externo
    print("💾 Exportando archivos finales...")
    exportar_para_notebook(df_completo, output_folder="data/processed")
    exportar_datos_finales(df_completo)
    
    
    GENERAR_GRAFICOS = False  # Cambiá a False para desactivar
    
    if GENERAR_GRAFICOS:
        print("🎨 Ejecutando generación de gráficos...")
        subprocess.run(["python", "main_graficos.py"])
        
        
        # =========================================================================
   # PARTE B: PIPELINE DE MACHINE LEARNING
   # =========================================================================
   print("\n--- INICIANDO PARTE B: SIMULACIÓN CON MACHINE LEARNING ---")

   # 1. Preparar datos
   X, y = prepare_data_for_ml(df_completo, ML_PARAMS['TARGET'], ML_PARAMS['FEATURES'])
   
   # 2. División de datos (temporal)
   # NOTA: Usamos train_test_split con shuffle=False para mantener el orden temporal
   from sklearn.model_selection import train_test_split
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
   print(pd.DataFrame(results))

   # 5. Visualizar Resultados del Mejor Modelo (Random Forest)
   print("\n🎨 Generando gráficos de resultados de ML...")
   rf_preds = rf_model.predict(X_test)
   plot_feature_importance(rf_model, ML_PARAMS['FEATURES'], PATHS['output_ml_plots'])
   plot_predictions_vs_actual(y_test, rf_preds, "Random Forest", PATHS['output_ml_plots'])



    print("\n🎉 Proceso completado exitosamente.")

if __name__ == "__main__":
    main()
