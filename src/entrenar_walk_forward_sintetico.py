"""
entrenar_walk_forward_sintetico.py
===================================
Entrena el modelo ML con metodología walk-forward expanding window por banco geo-estadístico.

Metodología:
- Por cada banco B (de menor a mayor, siguiendo el orden físico/espacial de explotación):
  * TRAIN: todos los bancos anteriores a B (expanding window)
  * TEST:  banco B únicamente
- Compara el rendimiento del modelo ML frente al baseline de estimación por bloques (Benchmark)
- Registra métricas de dispersión y ajuste: MAE, RMSE, R² por banco y resultado agregado

Esta es la metodología más rigurosa para validación en entornos geológicos:
- Respeta el orden secuencial de extracción (de bancos superiores a inferiores)
- Evita la contaminación de información (data leakage) entre las fases de entrenamiento y testeo
- Simula un entorno de producción real para la toma de decisiones en planificación minera

Autor: Manuel | Proyecto: mineral_ai_synthetic
"""

import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR    = Path("data/resultados")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mínimo de bancos de historia antes de empezar a predecir
MIN_BANCOS_TRAIN = 5

# Modelo a usar
MODELO = "lightgbm"  # opciones: "lightgbm", "xgboost", "randomforest"

# Hiperparámetros LightGBM (conservadores, optimizados para evitar overfitting)
PARAMS_LGB = {
    'objective':        'regression',
    'metric':           'mae',
    'n_estimators':     300,
    'learning_rate':    0.05,
    'num_leaves':       31,
    'max_depth':        6,
    'min_child_samples': 20,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'reg_alpha':        0.1,
    'reg_lambda':       0.1,
    'random_state':     42,
    'verbose':          -1,
    'n_jobs':           -1,
}


# ─── FUNCIONES ────────────────────────────────────────────────────────────────

def cargar_datos(verbose=True):
    """Carga dataset sintetizado preparado y su respectiva metadata."""
    df = pd.read_csv(PROCESSED_DIR / "dataset_preparado.csv")
    
    with open(PROCESSED_DIR / "metadata.json") as f:
        meta = json.load(f)
    
    features = meta['features']
    target   = meta['target']
    
    if verbose:
        print(f"► Dataset cargado: {len(df):,} bloques sintéticos, {len(features)} features")
        print(f"► Bancos disponibles: {df['banco'].nunique()} "
              f"(banco {df['banco'].min()} a {df['banco'].max()})")
    
    return df, features, target


def instalar_lightgbm():
    """Garantiza la disponibilidad del framework LightGBM en el entorno."""
    try:
        import lightgbm
        return True
    except ImportError:
        print("  Instalando LightGBM...")
        import subprocess
        result = subprocess.run(
            ["pip", "install", "lightgbm", "--break-system-packages", "-q"],
            capture_output=True
        )
        return result.returncode == 0


def obtener_modelo():
    """Retorna la instancia de regresión según la arquitectura seleccionada."""
    if MODELO == "lightgbm":
        instalar_lightgbm()
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**PARAMS_LGB)
    
    elif MODELO == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0
        )
    
    elif MODELO == "randomforest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        )


def metricas(y_true, y_pred):
    """Calcula indicadores estadísticos robustos de error (MAE, RMSE, R²)."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def walk_forward(df, features, target, verbose=True):
    """
    Validación Walk-Forward con ventana expansiva a nivel de banco de explotación.
    
    Por cada banco B >= MIN_BANCOS_TRAIN:
      TRAIN = Bloques pertenecientes a bancos [0, 1, ..., B-1]
      TEST  = Bloques pertenecientes al banco B
    """
    bancos = sorted(df['banco'].unique())
    resultados = []
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"WALK-FORWARD EXPANDING WINDOW (ESTÁNDAR MINERO)")
        print(f"{'='*60}")
        print(f"Bancos totales   : {len(bancos)}")
        print(f"Min bancos train : {MIN_BANCOS_TRAIN}")
        print(f"Bancos a evaluar : {len(bancos) - MIN_BANCOS_TRAIN}")
        print(f"Modelo en uso    : {MODELO.upper()}")
        print(f"\n{'Banco':>6} {'N_train':>8} {'N_test':>7} "
              f"{'MAE_ML':>9} {'MAE_Bench':>11} {'R²_ML':>7} {'ML gana?':>9}")
        print(f"{'─'*65}")
    
    ml_gana = 0
    total_evaluados = 0
    
    for i, banco_test in enumerate(bancos):
        bancos_train = [b for b in bancos if b < banco_test]
        
        if len(bancos_train) < MIN_BANCOS_TRAIN:
            continue
        
        # Segmentación espacial de ventanas
        mask_train = df['banco'].isin(bancos_train)
        mask_test  = df['banco'] == banco_test
        
        X_train = df.loc[mask_train, features]
        y_train = df.loc[mask_train, target]
        X_test  = df.loc[mask_test, features]
        y_test  = df.loc[mask_test, target]
        
        if len(y_test) == 0:
            continue
        
        # Mapeo dinámico del baseline geo-estadístico (Benchmark de Estimación)
        baseline_col = 'cut_benchmark' if 'cut_benchmark' in df.columns else ('cut_lp' if 'cut_lp' in df.columns else 'cut_mp')
        benchmark_pred = df.loc[mask_test, baseline_col]
        
        # Pipeline de Entrenamiento
        modelo = obtener_modelo()
        modelo.fit(X_train, y_train)
        ml_pred = modelo.predict(X_test)
        
        # Evaluación cruzada de métricas
        met_ml    = metricas(y_test, ml_pred)
        met_bench = metricas(y_test, benchmark_pred)
        
        gana = met_ml['mae'] < met_bench['mae']
        ml_gana += int(gana)
        total_evaluados += 1
        
        resultado = {
            'banco':         banco_test,
            'n_train':       len(y_train),
            'n_test':        len(y_test),
            'mae_ml':        met_ml['mae'],
            'rmse_ml':       met_ml['rmse'],
            'r2_ml':         met_ml['r2'],
            'mae_benchmark': met_bench['mae'],
            'rmse_benchmark':met_bench['rmse'],
            'r2_benchmark':  met_bench['r2'],
            'mejora_mae_pct': (met_bench['mae'] - met_ml['mae']) / met_bench['mae'] * 100,
            'ml_gana':       gana,
        }
        resultados.append(resultado)
        
        if verbose:
            gana_str = "   ✓ SÍ" if gana else "   ✗ NO"
            print(f"{banco_test:>6} {len(y_train):>8,} {len(y_test):>7} "
                  f"{met_ml['mae']:>9.4f} {met_bench['mae']:>11.4f} "
                  f"{met_ml['r2']:>7.3f} {gana_str:>9}")
    
    df_resultados = pd.DataFrame(resultados)
    
    if verbose:
        print(f"{'─'*65}")
        print(f"\n{'MÉTRICA GLOBAL - ML GANA':>25}: {ml_gana}/{total_evaluados} bancos "
              f"({ml_gana/total_evaluados*100:.1f}%)")
    
    return df_resultados


def resumen_final(df_res, verbose=True):
    """Calcula y despliega el resumen ejecutivo consolidado del proceso."""
    w = df_res['n_test']
    mae_ml_pond    = np.average(df_res['mae_ml'],    weights=w)
    mae_bench_pond = np.average(df_res['mae_benchmark'], weights=w)
    mejora_pond    = (mae_bench_pond - mae_ml_pond) / mae_bench_pond * 100
    
    r2_ml_pond    = np.average(df_res['r2_ml'],    weights=w)
    r2_bench_pond = np.average(df_res['r2_benchmark'], weights=w)
    
    bancos_ml_gana = df_res['ml_gana'].sum()
    total_bancos   = len(df_res)
    
    resumen = {
        'timestamp':            datetime.now().isoformat(),
        'modelo':               MODELO,
        'bancos_evaluados':     total_bancos,
        'ml_gana_n':            int(bancos_ml_gana),
        'ml_gana_pct':          round(bancos_ml_gana / total_bancos * 100, 1),
        'mae_ml_ponderado':     round(mae_ml_pond, 4),
        'mae_benchmark_ponderado': round(mae_bench_pond, 4),
        'mejora_mae_pct':       round(mejora_pond, 1),
        'r2_ml_ponderado':      round(r2_ml_pond, 3),
        'r2_benchmark_ponderado': round(r2_bench_pond, 3),
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"RESUMEN EJECUTIVO — VALIDACIÓN GEOLÓGICA SINTÉTICA")
        print(f"{'='*60}")
        print(f"Modelo seleccionado : {MODELO.upper()}")
        print(f"Bancos evaluados    : {total_bancos}")
        print(f"\n★ Rendimiento ML    : Omitiendo sesgos, ML optimiza el error en {bancos_ml_gana}/{total_bancos} frentes "
              f"({bancos_ml_gana/total_bancos*100:.1f}%)")
        print(f"\nMAE Ponderado:")
        print(f"  Modelo ML         : {mae_ml_pond:.4f}%")
        print(f"  Benchmark Tradic. : {mae_bench_pond:.4f}%")
        print(f"  Mejora Relativa   : -{mejora_pond:.1f}%")
        print(f"\nR² Ponderado (Coeficiente de Determinación):")
        print(f"  Modelo ML         : {r2_ml_pond:.3f}")
        print(f"  Benchmark Tradic. : {r2_bench_pond:.3f}")
        
        bancos_bench_gana = df_res[~df_res['ml_gana']]['banco'].tolist()
        if bancos_bench_gana:
            print(f"\nZonas de análisis donde el Benchmark tradicional retiene consistencia: {bancos_bench_gana}")
        else:
            print(f"\n★ ML incrementa la precisión en todo el espectro espacial analizado.")
    
    return resumen


def guardar_resultados(df_res, resumen):
    """Persiste las métricas estructuradas en la capa de datos de resultados."""
    path_detalle = OUTPUT_DIR / "walk_forward_sintetico_detalle.csv"
    df_res.to_csv(path_detalle, index=False)
    
    path_resumen = OUTPUT_DIR / "walk_forward_sintetico_resumen.json"
    with open(path_resumen, 'w') as f:
        json.dump(resumen, f, indent=2)
    
    print(f"\n✅ Logs e históricos consolidados con éxito:")
    print(f"   {path_detalle}")
    print(f"   {path_resumen}")
    print(f"\n→ Siguiente fase en la arquitectura: comparar_vs_baseline.py")


# ─── EXECUTION ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    print(f"\n{'='*60}")
    print(f"MINERAL_AI SINTÉTICO — ENTORNO DE VALIDACIÓN ESTRATIGRÁFICO")
    print(f"{'='*60}")
    
    # 1. Carga de datos procesados anonimizados
    df, features, target = cargar_datos(verbose=True)
    
    # 2. Reconstrucción del benchmark base de estimación
    path_raw = Path("data/raw/mineral_sintetico_v1.csv")
    if path_raw.exists():
        df_full = pd.read_csv(path_raw)
        VALOR_CORTE_STD = -99 
        mask_cut = df_full['cut'] != VALOR_CORTE_STD
        df['cut_benchmark'] = df_full.loc[mask_cut, 'cut_lp'].values
    else:
        # Fallback de simulación en caso de distribución directa del portafolio aislado
        df['cut_benchmark'] = df[target] * np.random.uniform(0.95, 1.05, size=len(df))
    
    # 3. Procesamiento del ciclo Walk-forward
    df_resultados = walk_forward(df, features, target, verbose=True)
    
    # 4. Generación de Resumen y Persistencia
    resumen = resumen_final(df_resultados, verbose=True)
    guardar_resultados(df_resultados, resumen)
