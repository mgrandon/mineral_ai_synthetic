"""
generar_dataset_sintetico.py
============================
Genera un dataset sintético de bloques de mineral de cobre que replica
la estructura real de un modelo de bloques minero (corto + mediano plazo).

Características del generador:
- Geometría 3D realista con bancos y continuidad espacial
- Litología y alteración con correlación espacial (no aleatoria pura)
- Leyes Cu simuladas con variograma simple (zonas ricas/pobres continuas)
- Cobertura de 'cut' real ~6% (fiel al mundo real)
- Variables de largo plazo tipo "Vulcan" con sesgo controlado
- Manejo correcto de valores nulos (-99 como en los datos reales)
- Compatible con el pipeline walk-forward ya desarrollado

Autor: Manuel + Claude | Proyecto: mineral_ai_synthetic
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ─── SEMILLA PARA REPRODUCIBILIDAD ───────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

# ─── PARÁMETROS DEL YACIMIENTO SINTÉTICO ─────────────────────────────────────
# Grilla de bloques (realista para mina open pit mediana)
N_X = 80        # bloques en dirección X
N_Y = 60        # bloques en dirección Y
N_BANCOS = 40   # bancos verticales (profundidad)
DIM_BLOQUE = 15 # metros por lado (bloques regulares)
ALTURA_BANCO = 15

COBERTURA_CUT_REAL = 0.057  # 5.7% bloques con ley real medida (igual al real)

# Coordenadas base del yacimiento (tipo coordenadas UTM genéricas)
X_BASE = 350_000
Y_BASE = 7_100_000
Z_BASE = 2_800   # metros sobre nivel del mar (mina de altura tipo Atacama)

# ─── TABLAS DE CATEGORÍAS (réplica fiel a las reales) ────────────────────────
# Litología: códigos enteros con significado geológico
LITOLOGIAS = {
    1: ('GRAN', 0.35),   # Granodiorita — 35% del yacimiento
    2: ('DIOP', 0.20),   # Diorita Pórfido
    3: ('BXCU', 0.15),   # Brecha Cuperífera (zona rica)
    4: ('ARGL', 0.12),   # Arcillita (zona alterada)
    5: ('TOBI', 0.10),   # Toba
    6: ('ANDF', 0.08),   # Andesita fracturada
}

# Alteración: correlacionada con litología
ALTERACIONES = {
    1: ('POTA', 0.25),   # Potásica (alta ley Cu)
    2: ('FILI', 0.30),   # Fílica
    3: ('PROP', 0.20),   # Propilítica (baja ley)
    4: ('ARGL', 0.15),   # Argílica
    5: ('SUPR', 0.10),   # Supergénica (Cu soluble alto)
}

# Zona mineral (relacionada con alteración y profundidad)
ZONAS_MINERAL = {
    1: 'OXID',   # Oxidado
    2: 'MIXTO',  # Mixto
    3: 'SULF',   # Sulfuro primario
    4: 'SECUND', # Sulfuro secundario
}


# ─── FUNCIONES DE SOPORTE ────────────────────────────────────────────────────

def simular_campo_espacial(nx, ny, nz, rango=15, nugget=0.1):
    """
    Simula un campo espacialmente correlacionado usando promedio móvil 3D.
    Replica la continuidad espacial de leyes en un yacimiento real.
    rango: radio de correlación en número de bloques
    nugget: varianza de ruido local (efecto pepita)
    """
    # Campo base aleatorio
    campo_base = rng.standard_normal((nx, ny, nz))
    
    # Suavizado gaussiano para simular variograma esférico
    from scipy.ndimage import gaussian_filter
    sigma = rango / 3.0
    campo_suave = gaussian_filter(campo_base, sigma=sigma)
    
    # Normalizar a [0, 1]
    campo_norm = (campo_suave - campo_suave.min()) / (campo_suave.max() - campo_suave.min())
    
    # Agregar nugget (variabilidad local)
    campo_final = (1 - nugget) * campo_norm + nugget * rng.random((nx, ny, nz))
    
    return campo_final


def asignar_litologia_espacial(campo_lito, nx, ny, nz):
    """
    Asigna litología con continuidad espacial usando el campo simulado.
    Las litologías no son aleatorias — forman dominios geológicos.
    """
    litos = np.zeros((nx, ny, nz), dtype=int)
    codigos = list(LITOLOGIAS.keys())
    proporciones = [LITOLOGIAS[k][1] for k in codigos]
    
    # Cuantiles para cada litología
    umbrales = np.cumsum([0] + proporciones)
    
    for i, cod in enumerate(codigos):
        mask = (campo_lito >= umbrales[i]) & (campo_lito < umbrales[i+1])
        litos[mask] = cod
    
    return litos


def calcular_ley_cu_realista(campo_cu, litos, alters, bancos_idx):
    """
    Calcula ley de Cu considerando:
    - Campo espacial base
    - Bonus por litología (brecha = más rica)
    - Bonus por alteración (potásica = más rica)
    - Tendencia con profundidad (supergénico en bancos altos)
    """
    # Ley base desde campo espacial (transformada lognormal)
    # Distribución típica: media ~0.5%, rango 0.1-2.5%
    ley_base = np.exp(campo_cu * 1.2 - 0.8)  # lognormal aprox
    ley_base = np.clip(ley_base, 0.05, 3.5)
    
    # Multiplicadores por litología
    mult_lito = np.ones_like(ley_base)
    mult_lito[litos == 3] = 1.8   # BXCU — brecha cuperífera, zona rica
    mult_lito[litos == 1] = 1.1   # GRAN — granodiorita, levemente enriquecida
    mult_lito[litos == 4] = 0.7   # ARGL — arcillita, zona pobre
    mult_lito[litos == 5] = 0.6   # TOBI — tobas, dilución
    
    # Multiplicadores por alteración
    mult_alter = np.ones_like(ley_base)
    mult_alter[alters == 1] = 1.5  # Potásica
    mult_alter[alters == 5] = 1.3  # Supergénica (Cu soluble)
    mult_alter[alters == 3] = 0.8  # Propilítica
    
    # Tendencia vertical (bancos superficiales = oxidado/mixto, profundos = sulfuro)
    # bancos_idx: 0 = más profundo, N_BANCOS-1 = más superficial
    factor_prof = 1 + 0.3 * (bancos_idx / N_BANCOS)  # levemente mayor en superficie
    
    ley_final = ley_base * mult_lito * mult_alter * factor_prof
    ley_final = np.clip(ley_final, 0.0, 4.0)
    
    return ley_final


def agregar_sesgo_vulcan(ley_real, bancos_idx, seed_local=123):
    """
    Simula la predicción tipo 'Vulcan' (geoestadística kriging).
    Vulcan tiene sesgo conocido:
    - Sobreestima en bancos profundos (~+5% promedio)
    - Subestima en zonas de alta ley (~-8%)
    - Error sistemático que varía por banco (como muestra la reconciliación histórica)
    Esto replica los patrones reales de la imagen de reconciliación 2004-2013.
    """
    rng_v = np.random.default_rng(seed_local)
    
    # Sesgo por banco (replica figura histórica: oscila ±5-10%)
    n_bancos = bancos_idx.max() + 1
    sesgo_banco = 0.05 * np.sin(np.linspace(0, 2*np.pi, n_bancos)) + \
                  rng_v.normal(0, 0.03, n_bancos)
    
    sesgo_local = sesgo_banco[bancos_idx]
    
    # Suavizado espacial (Vulcan no reproduce alta variabilidad local)
    suavizado = 0.15 * rng_v.normal(0, 1, ley_real.shape)
    
    # Regresión hacia la media (kriging suaviza extremos)
    media_global = ley_real.mean()
    regresion = 0.1 * (media_global - ley_real)
    
    ley_vulcan = ley_real * (1 + sesgo_local) + suavizado + regresion
    ley_vulcan = np.clip(ley_vulcan, 0.0, 4.0)
    
    return ley_vulcan


# ─── GENERADOR PRINCIPAL ──────────────────────────────────────────────────────

def generar_dataset(n_x=N_X, n_y=N_Y, n_bancos=N_BANCOS,
                    cobertura_cut=COBERTURA_CUT_REAL,
                    incluir_mediano_plazo=True,
                    verbose=True):
    """
    Genera el dataset completo de bloques sintéticos.
    
    Parámetros:
    -----------
    n_x, n_y, n_bancos: dimensiones de la grilla
    cobertura_cut: fracción de bloques con ley real medida
    incluir_mediano_plazo: genera también columnas _mp (mediano plazo)
    verbose: imprime resumen del proceso
    
    Retorna:
    --------
    DataFrame con estructura idéntica al dataset real
    """
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        print("Instalando scipy...")
        import subprocess
        subprocess.run(["pip", "install", "scipy", "--break-system-packages", "-q"])
        from scipy.ndimage import gaussian_filter

    n_total = n_x * n_y * n_bancos
    
    if verbose:
        print(f"{'='*60}")
        print(f"GENERADOR MINERAL_AI SINTÉTICO")
        print(f"{'='*60}")
        print(f"Grilla: {n_x} x {n_y} x {n_bancos} = {n_total:,} bloques")
        print(f"Cobertura cut real objetivo: {cobertura_cut:.1%}")
        print(f"Mediano plazo: {'Sí' if incluir_mediano_plazo else 'No'}")
        print()

    # ── 1. CAMPOS ESPACIALES BASE ─────────────────────────────────────────────
    if verbose: print("► Simulando campos espaciales...")
    
    campo_lito  = simular_campo_espacial(n_x, n_y, n_bancos, rango=20, nugget=0.05)
    campo_alter = simular_campo_espacial(n_x, n_y, n_bancos, rango=12, nugget=0.10)
    campo_cu    = simular_campo_espacial(n_x, n_y, n_bancos, rango=18, nugget=0.15)
    campo_fet   = simular_campo_espacial(n_x, n_y, n_bancos, rango=15, nugget=0.12)
    campo_as    = simular_campo_espacial(n_x, n_y, n_bancos, rango=10, nugget=0.20)

    # ── 2. LITOLOGÍA Y ALTERACIÓN ─────────────────────────────────────────────
    if verbose: print("► Asignando litología y alteración...")
    
    litos_3d  = asignar_litologia_espacial(campo_lito, n_x, n_y, n_bancos)
    
    # Alteración correlacionada con litología (no independiente)
    campo_alter_corr = 0.6 * campo_alter + 0.4 * campo_lito  # correlación parcial
    campo_alter_corr = (campo_alter_corr - campo_alter_corr.min()) / \
                       (campo_alter_corr.max() - campo_alter_corr.min())
    
    alters_3d = asignar_litologia_espacial(campo_alter_corr, n_x, n_y, n_bancos)
    # Recodificar a códigos de alteración (1-5)
    alters_3d = ((alters_3d - 1) % 5) + 1

    # ── 3. ÍNDICES DE BANCO ───────────────────────────────────────────────────
    # banco_idx: 0=más profundo, n_bancos-1=más superficial
    bancos_idx_3d = np.zeros((n_x, n_y, n_bancos), dtype=int)
    for b in range(n_bancos):
        bancos_idx_3d[:, :, b] = b

    # ── 4. LEYES ──────────────────────────────────────────────────────────────
    if verbose: print("► Calculando leyes de Cu y elementos asociados...")
    
    cut_real_3d = calcular_ley_cu_realista(campo_cu, litos_3d, alters_3d, bancos_idx_3d)
    cut_lp_3d   = agregar_sesgo_vulcan(cut_real_3d, bancos_idx_3d.flatten().reshape(n_x,n_y,n_bancos))

    # Cu soluble (correlacionado con cut total, mayor en zonas supergénicas)
    factor_cus = np.where(alters_3d == 5, 0.7, 0.15)  # supergénica vs sulfuro
    cus_lp_3d  = cut_lp_3d * factor_cus * (0.8 + 0.4 * rng.random((n_x, n_y, n_bancos)))
    cus_lp_3d  = np.clip(cus_lp_3d, 0.0, cut_lp_3d)

    # Fierro (correlacionado negativamente con Cu en algunas zonas)
    fet_base = campo_fet * 8 + 2  # rango 2-10%
    fet_lp_3d = fet_base * (1 + 0.1 * rng.standard_normal((n_x, n_y, n_bancos)))
    fet_lp_3d = np.clip(fet_lp_3d, 0.5, 15.0)

    # Arsénico (problema ambiental, correlacionado con ciertas litologías)
    as_base = campo_as * 200 + 10  # ppm, rango 10-210
    as_base[litos_3d == 4] *= 2.5  # ARGL tiene más As
    as_lp_3d = as_base * (1 + 0.15 * rng.standard_normal((n_x, n_y, n_bancos)))
    as_lp_3d = np.clip(as_lp_3d, 0.0, 500.0)

    # Densidad (función de litología principalmente)
    dens_base = {1: 2.7, 2: 2.75, 3: 2.65, 4: 2.5, 5: 2.4, 6: 2.8}
    densty_3d = np.vectorize(lambda l: dens_base.get(l, 2.65))(litos_3d)
    densty_3d += rng.normal(0, 0.05, (n_x, n_y, n_bancos))
    densty_3d = np.clip(densty_3d, 2.2, 3.2)

    # ── 5. COORDENADAS ────────────────────────────────────────────────────────
    if verbose: print("► Construyendo grilla de coordenadas...")
    
    xs = X_BASE + np.arange(n_x) * DIM_BLOQUE
    ys = Y_BASE + np.arange(n_y) * DIM_BLOQUE
    zs = Z_BASE - np.arange(n_bancos) * ALTURA_BANCO  # z decrece con profundidad

    # Meshgrid → arrays planos
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')
    cx = XX.flatten()
    cy = YY.flatten()
    cz = ZZ.flatten()

    # Número de banco (cota dividida en bins, tipo variable categórica ordinal)
    banco_num = (np.arange(n_bancos)).repeat(n_x * n_y)
    banco_num = bancos_idx_3d.flatten()

    # ── 6. APLANAR ARRAYS 3D ──────────────────────────────────────────────────
    litos_f    = litos_3d.flatten()
    alters_f   = alters_3d.flatten()
    cut_real_f = cut_real_3d.flatten()
    cut_lp_f   = cut_lp_3d.flatten()
    cus_lp_f   = cus_lp_3d.flatten()
    fet_lp_f   = fet_lp_3d.flatten()
    as_lp_f    = as_lp_3d.flatten()
    densty_f   = densty_3d.flatten()

    # ── 7. COBERTURA DE CUT REAL (~5.7%) ──────────────────────────────────────
    if verbose: print(f"► Asignando cobertura de ley real ({cobertura_cut:.1%})...")
    
    # La ley real NO es aleatoria — viene de sondajes y tronadura
    # Se concentra en bloques explotados (bancos medios, cerca de fases activas)
    # Simulamos esto: mayor probabilidad en bancos intermedios (ya explotados)
    peso_banco = np.exp(-0.5 * ((banco_num - n_bancos * 0.4) / (n_bancos * 0.3))**2)
    peso_banco = peso_banco / peso_banco.sum()
    
    n_con_cut = int(n_total * cobertura_cut)
    idx_con_cut = rng.choice(n_total, size=n_con_cut, replace=False, p=peso_banco)
    
    cut_columna = np.full(n_total, -99.0)  # -99 = sin dato (igual al real)
    cut_columna[idx_con_cut] = cut_real_f[idx_con_cut]

    # ── 8. MEDIANO PLAZO (si se pide) ─────────────────────────────────────────
    if incluir_mediano_plazo:
        if verbose: print("► Generando variables de mediano plazo...")
        
        # cut_mp: predicción Vulcan con sesgo distinto al LP (más suavizado)
        cut_mp_f = agregar_sesgo_vulcan(cut_real_f, banco_num, seed_local=456)
        # Más suavizado que LP (mediano plazo = menos detalle)
        from scipy.ndimage import gaussian_filter1d
        cut_mp_f = gaussian_filter1d(cut_mp_f.reshape(n_x*n_y, n_bancos), sigma=1.5).flatten()
        cut_mp_f = np.clip(cut_mp_f, 0.0, 4.0)
        
        cus_mp_f = cus_lp_f * (0.95 + 0.1 * rng.random(n_total))
        mot_lp_f = cut_lp_f * 0.003 * (0.5 + rng.random(n_total))  # Mo ~0.3% del Cu
        fet_mp_f = fet_lp_f * (0.98 + 0.05 * rng.random(n_total))
        as_mp_f  = as_lp_f  * (1.02 + 0.08 * rng.random(n_total))
    
    # ── 9. VARIABLES ADICIONALES ──────────────────────────────────────────────
    # ── ZONA MINERAL — LÓGICA REAL ROSARIO ───────────────────────────────────
    # Fuente: zona_mineral.xlsx — Script Rosario
    # Referencia: cut_lp_f = CUT (100% cobertura, como Vulcan)
    #             cus_lp_f = CUS_SU proxy (cobre soluble sulfúrico)
    #             cus_ci   = simulado (~82% de CUS_SU, relación lab típica)
    #
    # Códigos numéricos:
    #   LIX=20, OXI=30, MIX=40, SEC=50, PRI=80, PRIPY=100
    #
    # Lógica completa Rosario:
    # cut ≤ 0.2                          → LIX
    # cut > 0.2 y cus_su > 0:
    #   X = cus_su/cut
    #   X < 0.20        → SEC
    #   0.20 ≤ X < 0.45 → MIX
    #   X ≥ 0.45        → OXI
    # cut > 0.2 y cus_su ≤ 0 y cus_ci > 0:
    #   X = (1.2181*cus_ci - 0.0004)/cut → mismos umbrales
    # cut > 0.2 y cus_su ≤ 0 y cus_ci ≤ 0:
    #   cut ≥ 0.6 → SEC
    #   cut < 0.6 → LIX
    #   cut ≥ 0.3 → PRI   (zona sin soluble)
    #   cut < 0.3 → PRIPY

    # CUS_SU: usamos cus_lp_f (proxy sulfato, 100% cobertura)
    # CUS_CI: simulado como ~82% de CUS_SU (relación típica de laboratorio)
    cus_su = cus_lp_f.copy()
    cus_ci = cus_su * 0.82 * (0.9 + 0.2 * rng.random(n_total))
    cus_ci = np.clip(cus_ci, 0.0, cus_su * 1.1)

    cut_safe = np.where(cut_lp_f > 0, cut_lp_f, 0.001)  # evitar div/0
    zona_mineral = np.full(n_total, 50, dtype=int)        # default SEC

    # Gate inicial: cut ≤ 0.2 → LIX directo
    mask_lix_bajo = cut_lp_f <= 0.2
    zona_mineral = np.where(mask_lix_bajo, 20, zona_mineral)

    # Bloques con cut > 0.2
    mask_cut_ok = ~mask_lix_bajo

    # Caso 1: cut > 0.2 y CUS_SU > 0
    mask_su = mask_cut_ok & (cus_su > 0)
    X1 = np.where(mask_su, cus_su / cut_safe, 0.0)
    zona_mineral = np.where(mask_su & (X1 < 0.20),                50, zona_mineral)  # SEC
    zona_mineral = np.where(mask_su & (X1 >= 0.20) & (X1 < 0.45), 40, zona_mineral)  # MIX
    zona_mineral = np.where(mask_su & (X1 >= 0.45),               30, zona_mineral)  # OXI

    # Caso 2: cut > 0.2 y CUS_SU ≤ 0 y CUS_CI > 0
    mask_ci = mask_cut_ok & (cus_su <= 0) & (cus_ci > 0)
    X2 = np.where(mask_ci, (1.2181 * cus_ci - 0.0004) / cut_safe, 0.0)
    zona_mineral = np.where(mask_ci & (X2 < 0.20),                50, zona_mineral)  # SEC
    zona_mineral = np.where(mask_ci & (X2 >= 0.20) & (X2 < 0.45), 40, zona_mineral)  # MIX
    zona_mineral = np.where(mask_ci & (X2 >= 0.45),               30, zona_mineral)  # OXI

    # Caso 3: cut > 0.2 y CUS_SU ≤ 0 y CUS_CI ≤ 0 (zona sin soluble)
    mask_sin_sol = mask_cut_ok & (cus_su <= 0) & (cus_ci <= 0)
    zona_mineral = np.where(mask_sin_sol & (cut_lp_f >= 0.6), 50,  zona_mineral)  # SEC
    zona_mineral = np.where(mask_sin_sol & (cut_lp_f <  0.6), 20,  zona_mineral)  # LIX
    zona_mineral = np.where(mask_sin_sol & (cut_lp_f >= 0.3), 80,  zona_mineral)  # PRI
    zona_mineral = np.where(mask_sin_sol & (cut_lp_f <  0.3), 100, zona_mineral)  # PRIPY

    # Modelo (zona de extracción, tipo Rosario=1, Rosario Oeste=2)
    modelo_f = np.where(cx < X_BASE + n_x * DIM_BLOQUE * 0.6, 1, 2)

    # Fase (fases de minería 1-5, bancos más superficiales = fases más antiguas)
    fase_f = np.clip((banco_num / (n_bancos / 5)).astype(int) + 1, 1, 5)

    # ── 10. CONSTRUIR DATAFRAME ───────────────────────────────────────────────
    if verbose: print("► Ensamblando DataFrame...")
    
    df = pd.DataFrame({
        # Coordenadas y geometría
        'centroid_x': cx,
        'centroid_y': cy,
        'centroid_z': cz,
        'dim_x': DIM_BLOQUE,
        'dim_y': DIM_BLOQUE,
        'dim_z': ALTURA_BANCO,
        'volume': float(DIM_BLOQUE ** 2 * ALTURA_BANCO),
        'banco': banco_num,
        
        # Target: ley real (con -99 donde no hay dato)
        'cut': cut_columna,
        
        # Predicciones tipo Vulcan (100% cobertura)
        'cut_lp': cut_lp_f,
        'cus_lp': cus_lp_f,
        'fet_lp': fet_lp_f,
        'as_lp':  as_lp_f,
        'densty_lp': densty_f,
        
        # Geología
        'roca_lp':  litos_f,
        'roca_lp2': litos_f,   # versión alternativa (simplificada)
        'roca_cp':  litos_f,   # corto plazo usa misma lito
        'alter':    alters_f,
        'alt_lp':   alters_f,
        'alt_lp2':  alters_f,
        'alt_cp':   alters_f,
        
        # Variables complementarias
        'zona_mineral': zona_mineral,
        'modelo': modelo_f,
        'fase': fase_f,
        
        # Densidad real (solo donde hay muestras)
        'densidad': np.where(cut_columna != -99, densty_f, -99.0),
    })

    # Mediano plazo
    if incluir_mediano_plazo:
        df['cut_mp']  = cut_mp_f
        df['cus_mp']  = cus_mp_f
        df['mot_lp']  = mot_lp_f
        df['fet_mp']  = fet_mp_f
        df['as_mp']   = as_mp_f

    # ── 11. RESUMEN Y VALIDACIÓN ──────────────────────────────────────────────
    if verbose:
        print()
        print(f"{'='*60}")
        print(f"RESUMEN DEL DATASET GENERADO")
        print(f"{'='*60}")
        print(f"Total bloques        : {len(df):,}")
        print(f"Columnas             : {len(df.columns)}")
        
        mask_cut = df['cut'] != -99
        print(f"\nCobertura cut real   : {mask_cut.sum():,} bloques ({mask_cut.mean():.2%})")
        print(f"Ley Cu real — media  : {df.loc[mask_cut,'cut'].mean():.3f}%")
        print(f"Ley Cu real — std    : {df.loc[mask_cut,'cut'].std():.3f}%")
        print(f"Ley Cu real — rango  : [{df.loc[mask_cut,'cut'].min():.3f}, {df.loc[mask_cut,'cut'].max():.3f}]%")
        
        print(f"\nLey Cu Vulcan (LP):")
        print(f"  media              : {df['cut_lp'].mean():.3f}%")
        print(f"  MAE vs real        : {abs(df.loc[mask_cut,'cut'] - df.loc[mask_cut,'cut_lp']).mean():.4f}%")
        
        print(f"\nBancos               : {df['banco'].nunique()} ({df['banco'].min()} a {df['banco'].max()})")
        print(f"Litologías           : {sorted(df['roca_lp'].unique())}")
        print(f"Alteraciones         : {sorted(df['alter'].unique())}")
        
        print(f"\n✓ Dataset listo para pipeline walk-forward")
    
    return df


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    print("\n🔶 MINERAL_AI — GENERADOR DE DATOS SINTÉTICOS")
    print("   Replica estructura real: bloques 3D, litología,")
    print("   alteración, leyes Cu con continuidad espacial\n")
    
    # Generar dataset
    df = generar_dataset(
        n_x=N_X,
        n_y=N_Y,
        n_bancos=N_BANCOS,
        cobertura_cut=COBERTURA_CUT_REAL,
        incluir_mediano_plazo=True,
        verbose=True
    )
    
    # Guardar
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "mineral_sintetico_v1.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Archivo guardado: {output_path}")
    print(f"   Tamaño: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"\n→ Siguiente paso: preparar_dataset.py")
    print(f"→ Luego: entrenar_walk_forward.py")
