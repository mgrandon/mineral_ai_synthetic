"""
preparar_dataset.py
===================
Prepara el dataset sintético (o real) para el pipeline ML.

Tareas:
- Carga el CSV generado
- Manejo correcto de valores nulos (-99 → NaN)
- Filtra solo bloques con ley real medida (cut != -99)
- Encoding de variables categóricas (litología, alteración)
- Selección de features (las mismas 21-22 del modelo real)
- Separación X / y
- Guarda dataset listo para entrenar

Autor: Manuel + Claude | Proyecto: mineral_ai_synthetic
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────

INPUT_PATH  = Path("data/raw/mineral_sintetico_v1.csv")
OUTPUT_DIR  = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Valor nulo real en los datos mineros
NULL_VALUE = -99.0

# Features seleccionadas — réplica fiel al modelo real
# Excluye SIEMPRE variables con data leakage
FEATURES_LEAKAGE = [
    # Variables derivadas de cut (data leakage confirmado)
    'umet_cp', 'umet_lp', 'umet_mp',
    'mnzn', 'categ', 'categ_lp',
    # El target mismo
    'cut',
]

FEATURES_MODELO = [
    # Espaciales
    'centroid_x', 'centroid_y', 'centroid_z', 'banco',
    # Litología (categóricas → encoding)
    'roca_lp', 'roca_lp2', 'roca_cp',
    # Alteración (categóricas → encoding)
    'alter', 'alt_lp', 'alt_lp2', 'alt_cp',
    # Cobre soluble (proxy del Cu oxidado/supergénico)
    'cus_lp',
    # Fierro
    'fet_lp',
    # Arsénico
    'as_lp',
    # Densidad
    'densty_lp',
]

# Features de mediano plazo (opcionales — si existen en el CSV)
FEATURES_MP_OPCIONALES = [
    'cut_mp', 'cus_mp', 'fet_mp', 'as_mp', 'mot_lp',
]

TARGET = 'cut'

# Columnas categóricas que necesitan encoding
COLS_CATEGORICAS = [
    'roca_lp', 'roca_lp2', 'roca_cp',
    'alter', 'alt_lp', 'alt_lp2', 'alt_cp',
]


# ─── FUNCIONES ────────────────────────────────────────────────────────────────

def cargar_datos(path: Path, verbose=True) -> pd.DataFrame:
    """Carga el CSV con manejo robusto de tipos y nulos."""
    if verbose:
        print(f"► Cargando: {path}")
    
    df = pd.read_csv(path)
    
    if verbose:
        print(f"  Shape original: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    
    return df


def limpiar_nulos(df: pd.DataFrame, null_val=NULL_VALUE, verbose=True) -> pd.DataFrame:
    """
    Reemplaza el valor centinela -99 por NaN.
    
    IMPORTANTE: En los datos mineros reales, -99 significa 'sin dato'
    pero el default puede mentir — algunas variables usan -99 como valor
    válido (ej: coordenadas negativas no existen aquí, pero en otros
    datasets sí). Se hace columna por columna con criterio.
    """
    df = df.copy()
    
    # Columnas numéricas donde -99 siempre es nulo
    cols_nulas = [c for c in df.select_dtypes(include=[np.number]).columns
                  if c not in ['centroid_x', 'centroid_y', 'centroid_z',
                               'dim_x', 'dim_y', 'dim_z', 'volume', 'banco',
                               'modelo', 'fase', 'zona_mineral']]
    
    antes = df[cols_nulas].eq(null_val).sum().sum()
    df[cols_nulas] = df[cols_nulas].replace(null_val, np.nan)
    
    if verbose:
        print(f"► Nulos reemplazados: {antes:,} valores -99 → NaN")
        print(f"  Columnas afectadas: {(df[cols_nulas].isna().sum() > 0).sum()}")
    
    return df


def filtrar_bloques_con_cut(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """
    Filtra solo los bloques donde existe ley real medida.
    Estos son los únicos bloques usables para entrenar/validar.
    """
    mask = df[TARGET].notna()
    df_filtrado = df[mask].copy()
    
    if verbose:
        print(f"► Filtrado por cut real:")
        print(f"  Total bloques     : {len(df):,}")
        print(f"  Con cut real      : {mask.sum():,} ({mask.mean():.2%})")
        print(f"  Sin cut (excluidos): {(~mask).sum():,}")
    
    return df_filtrado


def agregar_features_mp(df: pd.DataFrame, verbose=True) -> list:
    """
    Agrega features de mediano plazo si están disponibles.
    Retorna la lista final de features a usar.
    """
    features_finales = FEATURES_MODELO.copy()
    
    features_mp_presentes = [f for f in FEATURES_MP_OPCIONALES if f in df.columns]
    
    if features_mp_presentes:
        features_finales.extend(features_mp_presentes)
        if verbose:
            print(f"► Features mediano plazo agregadas: {features_mp_presentes}")
    
    # Filtrar solo las que existen en el DataFrame
    features_finales = [f for f in features_finales if f in df.columns]
    
    if verbose:
        print(f"► Total features: {len(features_finales)}")
    
    return features_finales


def encoding_categoricas(df: pd.DataFrame, cols: list, verbose=True) -> pd.DataFrame:
    """
    Encoding de variables categóricas.
    Usa Label Encoding (enteros) — apropiado para LightGBM/XGBoost
    que manejan categorías numéricas nativamente.
    
    NO usa One-Hot Encoding — en modelos de árbol no mejora y aumenta
    dimensionalidad innecesariamente.
    """
    df = df.copy()
    cols_presentes = [c for c in cols if c in df.columns]
    
    for col in cols_presentes:
        # Convertir a int (ya son enteros en nuestro caso)
        df[col] = df[col].fillna(-1).astype(int)
    
    if verbose and cols_presentes:
        print(f"► Encoding categóricas: {cols_presentes}")
        for col in cols_presentes:
            print(f"  {col}: {sorted(df[col].unique())} ({df[col].nunique()} categorías)")
    
    return df


def imputar_nulos_features(df: pd.DataFrame, features: list, verbose=True) -> pd.DataFrame:
    """
    Imputa nulos en features numéricas con mediana por banco.
    
    Lógica: si falta cus_lp en un bloque, usar la mediana de los
    bloques del mismo banco es mejor que la mediana global
    (respeta la estructura espacial vertical).
    """
    df = df.copy()
    cols_con_nulos = [f for f in features
                      if f in df.columns and df[f].isna().any()
                      and f not in COLS_CATEGORICAS]
    
    if not cols_con_nulos:
        if verbose: print("► Sin nulos en features — no se requiere imputación")
        return df
    
    if verbose:
        print(f"► Imputando {len(cols_con_nulos)} columnas con nulos (mediana por banco):")
    
    for col in cols_con_nulos:
        n_nulos = df[col].isna().sum()
        # Mediana por banco
        mediana_banco = df.groupby('banco')[col].transform('median')
        # Fallback: mediana global
        mediana_global = df[col].median()
        df[col] = df[col].fillna(mediana_banco).fillna(mediana_global)
        
        if verbose:
            print(f"  {col}: {n_nulos:,} nulos imputados")
    
    return df


def validar_dataset(df: pd.DataFrame, features: list, verbose=True):
    """Validaciones básicas antes de entrenar."""
    errores = []
    
    # Sin nulos en features
    nulos_features = df[features].isna().sum()
    if nulos_features.any():
        errores.append(f"Features con nulos: {nulos_features[nulos_features > 0].to_dict()}")
    
    # Sin nulos en target
    if df[TARGET].isna().any():
        errores.append(f"Target con {df[TARGET].isna().sum()} nulos")
    
    # Mínimo de bloques por banco para walk-forward
    bloques_por_banco = df.groupby('banco').size()
    bancos_pocos = bloques_por_banco[bloques_por_banco < 10]
    if len(bancos_pocos) > 0:
        errores.append(f"Bancos con < 10 bloques: {bancos_pocos.to_dict()}")
    
    if errores:
        print("⚠️  ADVERTENCIAS:")
        for e in errores: print(f"   - {e}")
    else:
        if verbose: print("✓  Validación OK — dataset listo para entrenar")
    
    return len(errores) == 0


def resumen_estadistico(df: pd.DataFrame, features: list):
    """Imprime estadísticas clave del dataset preparado."""
    print(f"\n{'='*60}")
    print(f"ESTADÍSTICAS DEL DATASET PREPARADO")
    print(f"{'='*60}")
    print(f"Bloques para entrenamiento: {len(df):,}")
    print(f"Bancos disponibles        : {df['banco'].nunique()} "
          f"(banco {df['banco'].min()} a {df['banco'].max()})")
    print(f"\nTarget (cut real):")
    print(f"  Media  : {df[TARGET].mean():.4f}%")
    print(f"  Std    : {df[TARGET].std():.4f}%")
    print(f"  Mínimo : {df[TARGET].min():.4f}%")
    print(f"  Máximo : {df[TARGET].max():.4f}%")
    print(f"  Q25    : {df[TARGET].quantile(0.25):.4f}%")
    print(f"  Q75    : {df[TARGET].quantile(0.75):.4f}%")
    
    print(f"\nDistribución por banco (top 5 con más bloques):")
    top5 = df.groupby('banco').size().sort_values(ascending=False).head(5)
    for banco, n in top5.items():
        print(f"  Banco {banco:3d}: {n:,} bloques")
    
    print(f"\nFeatures ({len(features)}):")
    for f in features:
        print(f"  {f}")


# ─── PIPELINE PRINCIPAL ───────────────────────────────────────────────────────

def preparar(input_path=INPUT_PATH, verbose=True):
    """
    Pipeline completo de preparación.
    Retorna (X, y, df_completo, features).
    """
    print(f"\n{'='*60}")
    print(f"PREPARACIÓN DEL DATASET")
    print(f"{'='*60}\n")
    
    # 1. Cargar
    df = cargar_datos(input_path, verbose)
    
    # 2. Limpiar nulos
    df = limpiar_nulos(df, verbose=verbose)
    
    # 3. Filtrar bloques con cut real
    df = filtrar_bloques_con_cut(df, verbose)
    
    # 4. Encoding categóricas
    df = encoding_categoricas(df, COLS_CATEGORICAS, verbose)
    
    # 5. Agregar features MP si existen
    features = agregar_features_mp(df, verbose)
    
    # 6. Imputar nulos en features
    df = imputar_nulos_features(df, features, verbose)
    
    # 7. Validar
    print()
    ok = validar_dataset(df, features, verbose)
    
    if not ok:
        print("⚠️  Revisar advertencias antes de entrenar")
    
    # 8. Separar X / y
    X = df[features].copy()
    y = df[TARGET].copy()
    
    # 9. Resumen
    if verbose:
        resumen_estadistico(df, features)
    
    # 10. Guardar
    df.to_csv(OUTPUT_DIR / "dataset_preparado.csv", index=False)
    X.to_csv(OUTPUT_DIR / "X.csv", index=False)
    y.to_csv(OUTPUT_DIR / "y.csv", index=False, header=True)
    
    # Guardar metadata
    import json
    metadata = {
        'features': features,
        'target': TARGET,
        'n_bloques': len(df),
        'n_bancos': df['banco'].nunique(),
        'bancos': sorted(df['banco'].unique().tolist()),
        'cobertura_cut': COBERTURA_CUT_REAL_APROX,
    }
    
    with open(OUTPUT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Archivos guardados en: {OUTPUT_DIR}/")
    print(f"   dataset_preparado.csv | X.csv | y.csv | metadata.json")
    print(f"\n→ Siguiente paso: entrenar_walk_forward.py")
    
    return X, y, df, features


# ─── MAIN ─────────────────────────────────────────────────────────────────────

COBERTURA_CUT_REAL_APROX = 0.057

if __name__ == "__main__":
    X, y, df, features = preparar(verbose=True)
