"""
entrenar_walk_forward_sintetico.py
===================================
Entrena el modelo ML con metodología walk-forward expanding window por banco.

Metodología:
- Por cada banco B (de menor a mayor, orden temporal/espacial):
  * TRAIN: todos los bancos anteriores a B (expanding window)
  * TEST:  banco B únicamente
- Compara ML vs baseline Vulcan (cut_lp) en cada banco
- Registra MAE, RMSE, R² por banco y resultado agregado

Esta es la metodología más rigurosa para datos mineros:
- Respeta el orden natural de explotación (de superficie hacia abajo
  o de bancos más antiguos a más nuevos)
- Evita contaminación temporal entre train y test
- Replica exactamente la validación que ganó 40/40 bancos en el modelo real

Autor: Manuel + Claude | Proyecto: mineral_ai_synthetic
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

# Hiperparámetros LightGBM (conservadores, sin overfitting)
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
    """Carga dataset preparado y metadata."""
    df = pd.read_csv(PROCESSED_DIR / "dataset_preparado.csv")
    
    with open(PROCESSED_DIR / "metadata.json") as f:
        meta = json.load(f)
    
    features = meta['features']
    target   = meta['target']
    
    if verbose:
        print(f"► Dataset cargado: {len(df):,} bloques, {len(features)} features")
        print(f"► Bancos disponibles: {df['banco'].nunique()} "
              f"(banco {df['banco'].min()} a {df['banco'].max()})")
    
    return df, features, target


def instalar_lightgbm():
    """Instala LightGBM si no está disponible."""
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
    """Retorna instancia del modelo según configuración."""
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
    """Calcula MAE, RMSE, R²."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def walk_forward(df, features, target, verbose=True):
    """
    Walk-forward expanding window por banco.
    
    Por cada banco B >= MIN_BANCOS_TRAIN:
      TRAIN = bancos [0, 1, ..., B-1]
      TEST  = banco B
    
    Retorna DataFrame con resultados por banco.
    """
    bancos = sorted(df['banco'].unique())
    resultados = []
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"WALK-FORWARD EXPANDING WINDOW")
        print(f"{'='*60}")
        print(f"Bancos totales   : {len(bancos)}")
        print(f"Min bancos train : {MIN_BANCOS_TRAIN}")
        print(f"Bancos a evaluar : {len(bancos) - MIN_BANCOS_TRAIN}")
        print(f"Modelo           : {MODELO.upper()}")
        print(f"\n{'Banco':>6} {'N_train':>8} {'N_test':>7} "
              f"{'MAE_ML':>9} {'MAE_Vulcan':>11} {'R²_ML':>7} {'ML gana?':>9}")
        print(f"{'─'*65}")
    
    ml_gana = 0
    total_evaluados = 0
    
    for i, banco_test in enumerate(bancos):
        # Necesitamos al menos MIN_BANCOS_TRAIN bancos anteriores
        bancos_train = [b for b in bancos if b < banco_test]
        
        if len(bancos_train) < MIN_BANCOS_TRAIN:
            continue
        
        # Split train / test
        mask_train = df['banco'].isin(bancos_train)
        mask_test  = df['banco'] == banco_test
        
        X_train = df.loc[mask_train, features]
        y_train = df.loc[mask_train, target]
        X_test  = df.loc[mask_test, features]
        y_test  = df.loc[mask_test, target]
        
        if len(y_test) == 0:
            continue
        
        # Baseline Vulcan (cut_lp del test)
        vulcan_pred = df.loc[mask_test, 'cut_lp'] if 'cut_lp' in df.columns \
                      else df.loc[mask_test, 'cut_mp']
        
        # Entrenar ML
        modelo = obtener_modelo()
        modelo.fit(X_train, y_train)
        ml_pred = modelo.predict(X_test)
        
        # Métricas
        met_ml     = metricas(y_test, ml_pred)
        met_vulcan = metricas(y_test, vulcan_pred)
        
        gana = met_ml['mae'] < met_vulcan['mae']
        ml_gana += int(gana)
        total_evaluados += 1
        
        resultado = {
            'banco':         banco_test,
            'n_train':       len(y_train),
            'n_test':        len(y_test),
            'mae_ml':        met_ml['mae'],
            'rmse_ml':       met_ml['rmse'],
            'r2_ml':         met_ml['r2'],
            'mae_vulcan':    met_vulcan['mae'],
            'rmse_vulcan':   met_vulcan['rmse'],
            'r2_vulcan':     met_vulcan['r2'],
            'mejora_mae_pct': (met_vulcan['mae'] - met_ml['mae']) / met_vulcan['mae'] * 100,
            'ml_gana':       gana,
        }
        resultados.append(resultado)
        
        if verbose:
            gana_str = "✓ SÍ" if gana else "✗ NO"
            print(f"{banco_test:>6} {len(y_train):>8,} {len(y_test):>7} "
                  f"{met_ml['mae']:>9.4f} {met_vulcan['mae']:>11.4f} "
                  f"{met_ml['r2']:>7.3f} {gana_str:>9}")
    
    df_resultados = pd.DataFrame(resultados)
    
    if verbose:
        print(f"{'─'*65}")
        print(f"\n{'ML GANA':>20}: {ml_gana}/{total_evaluados} bancos "
              f"({ml_gana/total_evaluados*100:.1f}%)")
    
    return df_resultados


def resumen_final(df_res, verbose=True):
    """Calcula y muestra el resumen ejecutivo de resultados."""
    
    # MAE ponderado por número de bloques en test
    w = df_res['n_test']
    mae_ml_pond     = np.average(df_res['mae_ml'],     weights=w)
    mae_vulcan_pond = np.average(df_res['mae_vulcan'], weights=w)
    mejora_pond     = (mae_vulcan_pond - mae_ml_pond) / mae_vulcan_pond * 100
    
    r2_ml_pond     = np.average(df_res['r2_ml'],     weights=w)
    r2_vulcan_pond = np.average(df_res['r2_vulcan'], weights=w)
    
    bancos_ml_gana = df_res['ml_gana'].sum()
    total_bancos   = len(df_res)
    
    resumen = {
        'timestamp':         datetime.now().isoformat(),
        'modelo':            MODELO,
        'bancos_evaluados':  total_bancos,
        'ml_gana_n':         int(bancos_ml_gana),
        'ml_gana_pct':       round(bancos_ml_gana / total_bancos * 100, 1),
        'mae_ml_ponderado':  round(mae_ml_pond, 4),
        'mae_vulcan_ponderado': round(mae_vulcan_pond, 4),
        'mejora_mae_pct':    round(mejora_pond, 1),
        'r2_ml_ponderado':   round(r2_ml_pond, 3),
        'r2_vulcan_ponderado': round(r2_vulcan_pond, 3),
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"RESUMEN EJECUTIVO — WALK-FORWARD SINTÉTICO")
        print(f"{'='*60}")
        print(f"Modelo              : {MODELO.upper()}")
        print(f"Bancos evaluados    : {total_bancos}")
        print(f"\n★ ML gana           : {bancos_ml_gana}/{total_bancos} bancos "
              f"({bancos_ml_gana/total_bancos*100:.1f}%)")
        print(f"\nMAE ponderado:")
        print(f"  ML                : {mae_ml_pond:.4f}%")
        print(f"  Vulcan            : {mae_vulcan_pond:.4f}%")
        print(f"  Mejora ML vs Vulcan: -{mejora_pond:.1f}%")
        print(f"\nR² ponderado:")
        print(f"  ML                : {r2_ml_pond:.3f}")
        print(f"  Vulcan            : {r2_vulcan_pond:.3f}")
        
        # Bancos donde Vulcan gana (para análisis)
        bancos_vulcan_gana = df_res[~df_res['ml_gana']]['banco'].tolist()
        if bancos_vulcan_gana:
            print(f"\nBancos donde Vulcan gana: {bancos_vulcan_gana}")
        else:
            print(f"\n★ ML gana en TODOS los bancos")
    
    return resumen


def guardar_resultados(df_res, resumen):
    """Guarda resultados detallados y resumen."""
    # Resultados por banco
    path_detalle = OUTPUT_DIR / "walk_forward_sintetico_detalle.csv"
    df_res.to_csv(path_detalle, index=False)
    
    # Resumen ejecutivo
    path_resumen = OUTPUT_DIR / "walk_forward_sintetico_resumen.json"
    with open(path_resumen, 'w') as f:
        json.dump(resumen, f, indent=2)
    
    print(f"\n✅ Resultados guardados:")
    print(f"   {path_detalle}")
    print(f"   {path_resumen}")
    print(f"\n→ Siguiente paso: comparar_vs_baseline.py")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    print(f"\n{'='*60}")
    print(f"MINERAL_AI SINTÉTICO — WALK-FORWARD")
    print(f"{'='*60}")
    
    # 1. Cargar datos
    df, features, target = cargar_datos(verbose=True)
    
    # Necesitamos cut_lp en el dataset completo para el baseline Vulcan
    # Lo recuperamos del CSV original (tiene todos los bloques)
    df_full = pd.read_csv(Path("data/raw/mineral_sintetico_v1.csv"))
    mask_cut = df_full['cut'] != -99
    df['cut_lp'] = df_full.loc[mask_cut, 'cut_lp'].values
    
    # 2. Walk-forward
    df_resultados = walk_forward(df, features, target, verbose=True)
    
    # 3. Resumen
    resumen = resumen_final(df_resultados, verbose=True)
    
    # 4. Guardar
    guardar_resultados(df_resultados, resumen)