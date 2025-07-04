# src/train.py
# Modulo refactorizado final para soportar la plataforma de experimentacion.

import pandas as pd
import numpy as np
import logging
import os
import time
import warnings
from joblib import dump
import optuna

# Imports de Scikit-learn
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# Importamos nuestra configuracion centralizada y otras herramientas
from src import config, analysis_tools

logger = logging.getLogger('PipelineLogger')


def _get_model(model_name, hyperparams):
    """Crea una instancia de un modelo con sus hiperparametros."""
    if model_name == 'Random Forest':
        valid_params = {k: v for k, v in hyperparams.items() if k in RandomForestRegressor().get_params()}
        return RandomForestRegressor(**valid_params)
    if model_name == 'Gradient Boosting':
        valid_params = {k: v for k, v in hyperparams.items() if k in GradientBoostingRegressor().get_params()}
        return GradientBoostingRegressor(**valid_params)
    if model_name == 'Linear Regression':
        return LinearRegression()
    if model_name == 'MLP Neural Network':
        valid_params = {k: v for k, v in hyperparams.items() if k in MLPRegressor().get_params()}
        return MLPRegressor(**valid_params)
    raise ValueError(f"Modelo '{model_name}' no reconocido.")

def _entrenar_evaluar_un_fold(X_train, y_train, X_test, y_test, model_name, hyperparams):
    """Entrena y evalúa un único modelo para un solo fold."""
    start_time = time.perf_counter()
    model = _get_model(model_name, hyperparams)
    
    convergence_status = 'N/A'
    if hasattr(model, 'max_iter'):
        convergence_status = 'OK'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_train, y_train)
            if any(issubclass(warn.category, ConvergenceWarning) for warn in w):
                convergence_status = 'Not Converged'
    else:
        model.fit(X_train, y_train)

    preds_log = model.predict(X_test)
    end_time = time.perf_counter()
    
    results = {
        'R2_log': r2_score(y_test, preds_log),
        'RMSE_log': np.sqrt(mean_squared_error(y_test, preds_log)),
        'PBIAS_log': analysis_tools.pbias(y_test, pd.Series(preds_log)),
        'NSE_log': analysis_tools.nse(y_test, pd.Series(preds_log)),
        'training_time_secs': end_time - start_time,
        'convergence_status': convergence_status
    }
    return results, preds_log

def _objective_function(trial, X, y, model_name, cv_strategy, search_space):
    """
    Función que Optuna intentará optimizar.
    Define los hiperparámetros a probar, entrena el modelo y devuelve el score.
    """
    if model_name == 'Random Forest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *search_space['n_estimators']),
            'max_depth': trial.suggest_int('max_depth', *search_space['max_depth']),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', *search_space['min_samples_leaf']),
        }
    else: # Puedes añadir más modelos aquí si quieres
        raise NotImplementedError("La optimización Bayesiana solo está implementada para Random Forest.")

    model = _get_model(model_name, params)
    
    scores = []
    for train_idx, test_idx in cv_strategy.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        scores.append(r2_score(y_test, preds))
        
    return np.mean(scores) # Optuna maximizará el R2 promedio

def preparar_datos_modelo(df, feature_set):
    """Prepara el DataFrame seleccionando features y transformando el target."""
    logger.info("Preparando datos para el modelo...")
    target = config.TARGET_COLUMN
    features = [f for f in feature_set if f in df.columns]
    logger.info(f"El modelo será entrenado con {len(features)} features.")
    
    X = df[features]
    y = np.log1p(df[target])
    logger.info(f"Variable objetivo '{target}' transformada a escala logaritmica (log(1+y)).")
    return X, y, features

# --- FUNCIÓN ORQUESTADORA PRINCIPAL ---

def ejecutar_experimento_ml(df, exp_config, run_output_dir, cluster_id=None):
    """
    Función principal que orquesta el experimento de ML.
    SIN IMPUTACIÓN, con escalado correcto y guardado de todos los artefactos.
    """
    model_name = exp_config['model_name']
    logger.info(f"--- Iniciando Experimento de ML para '{model_name}' ---")

    X, y, features_finales = preparar_datos_modelo(df, exp_config['feature_set'])
    if X.empty:
        return None, None, None, None

    optimization_strategy = exp_config.get('optimization_strategy')
    
    final_model, summary_df, results_df, last_fold_data = None, None, None, None
    scaler = None # Se inicializa aquí para guardarlo al final

   

    if optimization_strategy == 'bayesian':
        # --- MODO OPTIMIZACIÓN BAYESIANA ---
        logger.info(f"Ejecutando en modo optimización: Bayesiana (Optuna) para {model_name}")
        
        # El escalado debe ocurrir antes si es necesario para el modelo
        X_processed = X.copy()
        if model_name in ['Linear Regression', 'MLP Neural Network']:
            scaler = StandardScaler()
            X_processed = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Creamos el estudio de Optuna
        study = optuna.create_study(direction='maximize')
        # Definimos la función objetivo con los datos y parámetros fijos
        objective = lambda trial: _objective_function(
            trial, X_processed, y, model_name, 
            TimeSeriesSplit(n_splits=5), 
            exp_config['search_space']
        )
        
        # Ejecutamos la optimización
        study.optimize(objective, n_trials=exp_config.get('n_trials', 50)) # 50 iteraciones por defecto

        logger.info(f"Optimización Bayesiana finalizada. Mejor R2 (cross-validated): {study.best_value:.4f}")
        logger.info(f"Mejores hiperparámetros: {study.best_params}")
        
        final_model = _get_model(model_name, study.best_params)
        final_model.fit(X_processed, y)
        
        results_df = pd.DataFrame([{
            'fold': 'N/A', # <-- SE AÑADE ESTA LÍNEA
            'model': model_name, 
            'base_experiment_id': exp_config.get('original_id', exp_config['id']),
            'cluster_id': cluster_id, 
            'R2_log': grid_search.best_score_, # <-- Usamos el resultado REAL de la CV
            'RMSE_log': np.nan, # No tenemos un RMSE promediado de la CV
            'PBIAS_log': np.nan, # No tenemos un PBIAS promediado de la CV
            'hyperparameters': str(grid_search.best_params_)
        }])
        
        summary_df = results_df.drop(columns=['fold']).set_index('model')
        last_fold_data = {'test_indices': X.index, 'y_test': y, 'predictions_log': preds_log, 'features': features_finales}



    elif optimization_strategy == 'grid_search':
        # --- MODO GRIDSEARCH ---
        logger.info(f"Ejecutando en modo optimización: GridSearchCV para {model_name}")
        
        X_processed = X.copy()
        if model_name in ['Linear Regression', 'MLP Neural Network']:
            scaler = StandardScaler()
            X_processed = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        cv_strategy = TimeSeriesSplit(n_splits=exp_config.get('time_series_splits', 5))
        base_model = _get_model(model_name, {})
        grid_search = GridSearchCV(estimator=base_model, param_grid=exp_config['param_grid'], cv=cv_strategy, scoring='r2', n_jobs=-1, verbose=2)
        grid_search.fit(X_processed, y)

        logger.info(f"GridSearchCV finalizado. Mejor R2 (cross-validated): {grid_search.best_score_:.4f}")
        logger.info(f"Mejores hiperparámetros: {grid_search.best_params_}")
        
        final_model = grid_search.best_estimator_
        preds_log = final_model.predict(X_processed)
        
        results_df = pd.DataFrame([{
            'fold': 'N/A', # <-- SE AÑADE ESTA LÍNEA
            'model': model_name, 
            'base_experiment_id': exp_config.get('original_id', exp_config['id']),
            'cluster_id': cluster_id, 
            'R2_log': grid_search.best_score_, # <-- Usamos el resultado REAL de la CV
            'RMSE_log': np.nan, # No tenemos un RMSE promediado de la CV
            'PBIAS_log': np.nan, # No tenemos un PBIAS promediado de la CV
            'hyperparameters': str(grid_search.best_params_)
        }])
        summary_df = results_df.drop(columns=['fold']).set_index('model')
        last_fold_data = {'test_indices': X.index, 'y_test': y, 'predictions_log': preds_log, 'features': features_finales}
        
    else:
        # --- MODO DE ENTRENAMIENTO NORMAL ---
        logger.info(f"Ejecutando en modo de entrenamiento de modelo único para {model_name}")
        cv_strategy = TimeSeriesSplit(n_splits=exp_config.get('time_series_splits', 5))
        
        if exp_config.get('generate_eda_plots', False):
            eda_plot_path = os.path.join(run_output_dir, "plots", "EDA")
            os.makedirs(eda_plot_path, exist_ok=True)
            analysis_tools.plot_cv_splits(cv_strategy, X, y, eda_plot_path, os.path.basename(run_output_dir), exp_config['validation_strategy'])

        all_results = []
        if model_name in ['Linear Regression', 'MLP Neural Network']:
            scaler = StandardScaler()

        for i, (train_idx, test_idx) in enumerate(cv_strategy.split(X, y)):
            X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            X_train_processed, X_test_processed = X_train_raw.copy(), X_test_raw.copy()
            if scaler:
                X_train_processed = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X.columns)
                X_test_processed = pd.DataFrame(scaler.transform(X_test_raw), columns=X.columns)

            fold_results, preds_log_fold = _entrenar_evaluar_un_fold(X_train_processed, y_train, X_test_processed, y_test, model_name, exp_config.get('hyperparameters', {}))
            
            res = {'fold': i + 1, 'model': model_name, 'base_experiment_id': exp_config.get('original_id', exp_config['id'])}
            if cluster_id is not None: res['cluster_id'] = cluster_id
            res.update(fold_results)
            all_results.append(res)
            
            if i == cv_strategy.get_n_splits(X,y) - 1:
                last_fold_data = {
                    'test_indices': X_test_raw.index, # <-- Usamos el índice de X_test
                    'y_test': y_test, 
                    'predictions_log': preds_log_fold, 
                    'features': features_finales
                }
        
        results_df = pd.DataFrame(all_results)
        
     # 1. Resumen para métricas NUMÉRICAS
        metricas_numericas = [col for col in ['R2_log', 'RMSE_log', 'PBIAS_log', 'training_time_secs'] if col in results_df.columns]
        if metricas_numericas:
            summary_df = results_df.groupby('model')[metricas_numericas].agg(['mean', 'std']).round(4)
            logger.info(f"\n--- Resumen de Métricas Numéricas para '{exp_config['id']}' ---")
            logger.info(f"\n{summary_df.to_string()}")
        else:
            summary_df = pd.DataFrame() # DF vacío si no hay métricas

        # 2. Resumen para métricas CATEGÓRICAS (como el estado de convergencia)
        if 'convergence_status' in results_df.columns:
            convergence_summary = results_df.groupby('model')['convergence_status'].value_counts().unstack(fill_value=0)
            logger.info(f"\n--- Resumen de Convergencia para '{exp_config['id']}' ---")
            logger.info(f"\n{convergence_summary.to_string()}")
            
        final_model = _get_model(model_name, exp_config.get('hyperparameters', {}))
        
        X_final_processed = X.copy()
        if scaler:
            # Re-ajustamos el scaler con TODOS los datos para el modelo final
            X_final_processed = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        final_model.fit(X_final_processed, y)

    # --- Guardado unificado de artefactos ---
    if final_model:
        logger.info("--- Guardando artefactos de predicción... ---")
        artefactos_de_prediccion = {'model': final_model, 'scaler': scaler, 'features_finales': features_finales}
        model_filename = "prediction_artifacts.joblib"
        ruta_guardado = os.path.join(run_output_dir, "models", model_filename)
        dump(artefactos_de_prediccion, ruta_guardado)
        logger.info(f"Artefactos de predicción guardados en '{ruta_guardado}'")

    return final_model, summary_df, results_df, last_fold_data




def generar_reporte_rendimiento_espacial(df_datos_original, last_fold_data, output_path, run_id):
    """
    Evalúa el rendimiento del modelo por cuenca y por estación y guarda los resultados.
    """
    logger.info("Generando reporte de rendimiento espacial (por cuenca y estación)...")

    # 1. Preparar los datos: Unir predicciones con metadatos
    # Usamos los índices del fold de prueba para seleccionar las filas correspondientes
    df_predicciones = df_datos_original.loc[last_fold_data['test_indices']].copy()
    # Añadimos los valores reales y predichos del fold
    df_predicciones['y_test'] = last_fold_data['y_test']
    df_predicciones['predictions_log'] = last_fold_data['predictions_log']

    # 2. Calcular métricas por CUENCA
    resumen_por_cuenca = df_predicciones.groupby('basin').apply(analysis_tools.calcular_metricas_grupo).dropna().round(3)
    
    if not resumen_por_cuenca.empty:
        ruta_cuenca = os.path.join(output_path, f"{run_id}_rendimiento_por_cuenca.csv")
        resumen_por_cuenca.to_csv(ruta_cuenca)
        logger.info(f"  - Reporte por cuenca guardado en: {ruta_cuenca}")

    # 3. Calcular métricas por ESTACIÓN
    resumen_por_estacion = df_predicciones.groupby('station_code').apply(analysis_tools.calcular_metricas_grupo).dropna().round(3)

    if not resumen_por_estacion.empty:
        ruta_estacion = os.path.join(output_path, f"{run_id}_rendimiento_por_estacion.csv")
        resumen_por_estacion.to_csv(ruta_estacion)
        logger.info(f"  - Reporte por estación guardado en: {ruta_estacion}")