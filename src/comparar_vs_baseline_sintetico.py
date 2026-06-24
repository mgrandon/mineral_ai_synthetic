"""
comparar_vs_baseline_sintetico.py
==================================
Genera visualizaciones comparativas ML vs Vulcan (baseline geoestadístico).

Gráficos generados:
1. MAE por banco — ML vs Vulcan (líneas)
2. Mejora % por banco (barras, verde=ML gana, rojo=Vulcan gana)
3. R² por banco — ML vs Vulcan
4. Scatter: predicción vs real (ML y Vulcan lado a lado)
5. Distribución de errores (histograma)
6. Resumen ejecutivo visual (para README/portafolio)

Autor: Manuel + Claude | Proyecto: mineral_ai_synthetic
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────

RESULTS_DIR   = Path("data/resultados")
PROCESSED_DIR = Path("data/processed")
RAW_DIR       = Path("data/raw")
OUTPUT_DIR    = Path("data/figuras")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Paleta de colores — profesional, apta para presentaciones mineras
COLOR_ML     = '#1a6eb5'   # azul corporativo
COLOR_VULCAN = '#c0392b'   # rojo Vulcan
COLOR_VERDE  = '#27ae60'   # mejora positiva
COLOR_FONDO  = '#f8f9fa'
COLOR_GRID   = '#dee2e6'

FIGSIZE_WIDE = (14, 5)
FIGSIZE_TALL = (14, 10)
DPI = 150


# ─── FUNCIONES DE SOPORTE ────────────────────────────────────────────────────

def cargar_resultados():
    """Carga resultados del walk-forward y datos originales."""
    df_res = pd.read_csv(RESULTS_DIR / "walk_forward_sintetico_detalle.csv")
    
    with open(RESULTS_DIR / "walk_forward_sintetico_resumen.json") as f:
        resumen = json.load(f)
    
    # Dataset con predicciones para scatter
    df_proc = pd.read_csv(PROCESSED_DIR / "dataset_preparado.csv")
    df_raw  = pd.read_csv(RAW_DIR / "mineral_sintetico_v1.csv")
    
    # Agregar cut_lp al dataset procesado
    mask_cut = df_raw['cut'] != -99
    df_proc['cut_lp'] = df_raw.loc[mask_cut, 'cut_lp'].values
    
    print(f"► Resultados cargados: {len(df_res)} bancos evaluados")
    print(f"► Resumen: ML gana {resumen['ml_gana_n']}/{resumen['bancos_evaluados']} bancos")
    
    return df_res, resumen, df_proc


def estilo_grafico(ax, titulo, xlabel, ylabel):
    """Aplica estilo consistente a todos los gráficos."""
    ax.set_title(titulo, fontsize=13, fontweight='bold', pad=12, color='#2c3e50')
    ax.set_xlabel(xlabel, fontsize=10, color='#555')
    ax.set_ylabel(ylabel, fontsize=10, color='#555')
    ax.set_facecolor(COLOR_FONDO)
    ax.grid(True, color=COLOR_GRID, linewidth=0.8, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLOR_GRID)
    ax.spines['bottom'].set_color(COLOR_GRID)
    ax.tick_params(colors='#555', labelsize=9)


# ─── GRÁFICOS INDIVIDUALES ────────────────────────────────────────────────────

def grafico_mae_por_banco(df_res, ax=None):
    """MAE por banco: ML vs Vulcan."""
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    ax.plot(df_res['banco'], df_res['mae_ml'],
            color=COLOR_ML, linewidth=2.2, marker='o', markersize=4,
            label='ML (LightGBM)', zorder=3)
    ax.plot(df_res['banco'], df_res['mae_vulcan'],
            color=COLOR_VULCAN, linewidth=2.2, marker='s', markersize=4,
            linestyle='--', label='Vulcan (Kriging)', zorder=3)
    
    # Área entre curvas (zona de ventaja ML)
    ax.fill_between(df_res['banco'], df_res['mae_ml'], df_res['mae_vulcan'],
                    where=df_res['mae_ml'] < df_res['mae_vulcan'],
                    alpha=0.12, color=COLOR_VERDE, label='Ventaja ML')
    
    estilo_grafico(ax, 'MAE por Banco — ML vs Vulcan',
                   'Número de Banco', 'MAE (% Cu)')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_ylim(bottom=0)
    
    return ax


def grafico_mejora_por_banco(df_res, ax=None):
    """Mejora % por banco (barras)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    colores = [COLOR_VERDE if g else COLOR_VULCAN for g in df_res['ml_gana']]
    
    bars = ax.bar(df_res['banco'], df_res['mejora_mae_pct'],
                  color=colores, alpha=0.85, edgecolor='white', linewidth=0.5)
    
    ax.axhline(y=0, color='#333', linewidth=1.2, zorder=5)
    ax.axhline(y=df_res['mejora_mae_pct'].mean(), color=COLOR_ML,
               linewidth=1.5, linestyle=':', alpha=0.8,
               label=f"Promedio: {df_res['mejora_mae_pct'].mean():.1f}%")
    
    # Etiqueta en barras extremas
    max_idx = df_res['mejora_mae_pct'].idxmax()
    ax.annotate(f"{df_res.loc[max_idx,'mejora_mae_pct']:.1f}%",
                xy=(df_res.loc[max_idx,'banco'], df_res.loc[max_idx,'mejora_mae_pct']),
                xytext=(0, 5), textcoords='offset points',
                fontsize=8, ha='center', color=COLOR_VERDE, fontweight='bold')
    
    patch_ml     = mpatches.Patch(color=COLOR_VERDE, alpha=0.85, label='ML gana')
    patch_vulcan = mpatches.Patch(color=COLOR_VULCAN, alpha=0.85, label='Vulcan gana')
    
    estilo_grafico(ax, 'Mejora MAE por Banco (% reducción de error)',
                   'Número de Banco', 'Mejora MAE (%)')
    ax.legend(handles=[patch_ml, patch_vulcan], fontsize=9, framealpha=0.9)
    
    return ax


def grafico_r2_por_banco(df_res, ax=None):
    """R² por banco: ML vs Vulcan."""
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    ax.plot(df_res['banco'], df_res['r2_ml'],
            color=COLOR_ML, linewidth=2.2, marker='o', markersize=4,
            label='ML (LightGBM)', zorder=3)
    ax.plot(df_res['banco'], df_res['r2_vulcan'],
            color=COLOR_VULCAN, linewidth=2.2, marker='s', markersize=4,
            linestyle='--', label='Vulcan (Kriging)', zorder=3)
    
    ax.fill_between(df_res['banco'], df_res['r2_ml'], df_res['r2_vulcan'],
                    where=df_res['r2_ml'] > df_res['r2_vulcan'],
                    alpha=0.12, color=COLOR_VERDE)
    
    ax.axhline(y=0.9, color='#aaa', linewidth=1, linestyle=':',
               label='R²=0.90 (referencia)')
    
    estilo_grafico(ax, 'R² por Banco — ML vs Vulcan',
                   'Número de Banco', 'R²')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_ylim(0.5, 1.02)
    
    return ax


def grafico_scatter_prediccion(df_proc, ax_ml=None, ax_vul=None):
    """Scatter: predicción vs real para ML y Vulcan."""
    
    # Reconstruir predicciones ML usando el modelo entrenado en todos los datos
    # (para el scatter usamos una predicción representativa, no walk-forward)
    from sklearn.ensemble import RandomForestRegressor
    
    with open(PROCESSED_DIR / "metadata.json") as f:
        meta = json.load(f)
    features = meta['features']
    
    X = df_proc[features]
    y = df_proc['cut']
    
    # Train simple 80/20 para scatter ilustrativo
    n = len(df_proc)
    idx_train = df_proc.index[:int(n * 0.8)]
    idx_test  = df_proc.index[int(n * 0.8):]
    
    from lightgbm import LGBMRegressor
    model = LGBMRegressor(n_estimators=200, random_state=42, verbose=-1)
    model.fit(X.loc[idx_train], y.loc[idx_train])
    ml_pred = model.predict(X.loc[idx_test])
    
    y_test       = y.loc[idx_test]
    vulcan_test  = df_proc.loc[idx_test, 'cut_lp']
    
    if ax_ml is None or ax_vul is None:
        fig, (ax_ml, ax_vul) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    
    lim_min = min(y_test.min(), ml_pred.min(), vulcan_test.min()) * 0.9
    lim_max = max(y_test.max(), ml_pred.max(), vulcan_test.max()) * 1.05
    
    for ax, pred, color, titulo in [
        (ax_ml,  ml_pred,      COLOR_ML,     'ML (LightGBM)'),
        (ax_vul, vulcan_test,  COLOR_VULCAN, 'Vulcan (Kriging)'),
    ]:
        from sklearn.metrics import mean_absolute_error, r2_score
        mae = mean_absolute_error(y_test, pred)
        r2  = r2_score(y_test, pred)
        
        ax.scatter(y_test, pred, alpha=0.3, s=12, color=color, edgecolors='none')
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                'k--', linewidth=1.2, alpha=0.6, label='Predicción perfecta')
        
        ax.text(0.05, 0.92, f'MAE = {mae:.4f}%\nR² = {r2:.3f}',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor=color, alpha=0.9),
                verticalalignment='top', color=color, fontweight='bold')
        
        estilo_grafico(ax, f'Predicción vs Real — {titulo}',
                       'Ley Cu Real (%)', 'Ley Cu Predicha (%)')
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.legend(fontsize=8)
    
    return ax_ml, ax_vul


def grafico_resumen_ejecutivo(resumen, df_res):
    """
    Panel resumen ejecutivo — este es el gráfico principal del portafolio.
    Una sola imagen que cuenta toda la historia.
    """
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('white')
    
    gs = GridSpec(3, 3, figure=fig,
                  hspace=0.45, wspace=0.35,
                  top=0.88, bottom=0.08, left=0.07, right=0.96)
    
    # ── Título principal
    fig.text(0.5, 0.95,
             'mineral_ai — ML vs Geoestadística (Vulcan) en Predicción de Ley Cu',
             ha='center', va='top', fontsize=15, fontweight='bold', color='#2c3e50')
    fig.text(0.5, 0.91,
             'Validación Walk-Forward Expanding Window por Banco | Datos Sintéticos Calibrados',
             ha='center', va='top', fontsize=10, color='#7f8c8d')
    
    # ── KPIs grandes (fila superior)
    kpis = [
        (f"{resumen['ml_gana_n']}/{resumen['bancos_evaluados']}",
         "Bancos donde\nML supera Vulcan", COLOR_VERDE),
        (f"-{resumen['mejora_mae_pct']}%",
         "Reducción de\nerror MAE", COLOR_ML),
        (f"{resumen['r2_ml_ponderado']:.3f}",
         "R² ponderado\ndel modelo ML", COLOR_ML),
    ]
    
    for i, (valor, label, color) in enumerate(kpis):
        ax_kpi = fig.add_subplot(gs[0, i])
        ax_kpi.set_facecolor(COLOR_FONDO)
        ax_kpi.text(0.5, 0.58, valor,
                    ha='center', va='center', fontsize=28,
                    fontweight='bold', color=color,
                    transform=ax_kpi.transAxes)
        ax_kpi.text(0.5, 0.18, label,
                    ha='center', va='center', fontsize=9.5,
                    color='#555', transform=ax_kpi.transAxes)
        ax_kpi.set_xlim(0,1); ax_kpi.set_ylim(0,1)
        ax_kpi.axis('off')
        rect = plt.Rectangle((0.05, 0.05), 0.90, 0.90,
                              fill=True, facecolor=COLOR_FONDO,
                              edgecolor=color, linewidth=2,
                              transform=ax_kpi.transAxes, clip_on=False)
        ax_kpi.add_patch(rect)
    
    # ── MAE por banco (fila media, span 2 cols)
    ax_mae = fig.add_subplot(gs[1, :2])
    grafico_mae_por_banco(df_res, ax=ax_mae)
    ax_mae.set_title('MAE por Banco', fontsize=11, fontweight='bold', color='#2c3e50')
    
    # ── Mejora % (fila media, última col)
    ax_mej = fig.add_subplot(gs[1, 2])
    colores = [COLOR_VERDE if g else COLOR_VULCAN for g in df_res['ml_gana']]
    ax_mej.bar(df_res['banco'], df_res['mejora_mae_pct'],
               color=colores, alpha=0.85, edgecolor='white', linewidth=0.3)
    ax_mej.axhline(0, color='#333', linewidth=0.8)
    estilo_grafico(ax_mej, 'Mejora % por Banco',
                   'Banco', 'Mejora MAE (%)')
    
    # ── R² por banco (fila inferior, span 2 cols)
    ax_r2 = fig.add_subplot(gs[2, :2])
    grafico_r2_por_banco(df_res, ax=ax_r2)
    ax_r2.set_title('R² por Banco', fontsize=11, fontweight='bold', color='#2c3e50')
    
    # ── Tabla comparativa (fila inferior, última col)
    ax_tab = fig.add_subplot(gs[2, 2])
    ax_tab.axis('off')
    ax_tab.set_facecolor(COLOR_FONDO)
    
    tabla_data = [
        ['Métrica', 'ML', 'Vulcan'],
        ['MAE pond.', f"{resumen['mae_ml_ponderado']:.4f}%",
                      f"{resumen['mae_vulcan_ponderado']:.4f}%"],
        ['R² pond.',  f"{resumen['r2_ml_ponderado']:.3f}",
                      f"{resumen['r2_vulcan_ponderado']:.3f}"],
        ['Bancos\nganados', f"{resumen['ml_gana_n']}/35", "0/35"],
        ['Mejora\nMAE', f"-{resumen['mejora_mae_pct']}%", "—"],
    ]
    
    tabla = ax_tab.table(cellText=tabla_data[1:],
                         colLabels=tabla_data[0],
                         loc='center', cellLoc='center')
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.scale(1.1, 1.8)
    
    # Colorear encabezados
    for j in range(3):
        tabla[0, j].set_facecolor('#2c3e50')
        tabla[0, j].set_text_props(color='white', fontweight='bold')
    
    # Colorear columna ML de azul suave
    for i in range(1, len(tabla_data)):
        tabla[i, 1].set_facecolor('#e8f4fd')
        tabla[i, 1].set_text_props(color=COLOR_ML, fontweight='bold')
    
    ax_tab.set_title('Comparativa Final', fontsize=11,
                     fontweight='bold', color='#2c3e50', pad=8)
    
    # ── Firma
    fig.text(0.99, 0.01,
             'mineral_ai | github.com/tu-usuario/mineral_ai_synthetic',
             ha='right', va='bottom', fontsize=7.5, color='#aaa',
             style='italic')
    
    return fig


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    print(f"\n{'='*60}")
    print(f"MINERAL_AI — COMPARACIÓN ML vs VULCAN")
    print(f"{'='*60}\n")
    
    # 1. Cargar resultados
    df_res, resumen, df_proc = cargar_resultados()
    
    # 2. Gráfico resumen ejecutivo (el más importante — va en README)
    print("\n► Generando resumen ejecutivo...")
    fig_resumen = grafico_resumen_ejecutivo(resumen, df_res)
    path_resumen = OUTPUT_DIR / "resumen_ejecutivo.png"
    fig_resumen.savefig(path_resumen, dpi=DPI, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
    plt.close(fig_resumen)
    print(f"  ✓ {path_resumen}")
    
    # 3. MAE por banco
    print("► Generando MAE por banco...")
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE, facecolor='white')
    grafico_mae_por_banco(df_res, ax)
    path_mae = OUTPUT_DIR / "mae_por_banco.png"
    fig.savefig(path_mae, dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ {path_mae}")
    
    # 4. Mejora por banco
    print("► Generando mejora por banco...")
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE, facecolor='white')
    grafico_mejora_por_banco(df_res, ax)
    path_mej = OUTPUT_DIR / "mejora_por_banco.png"
    fig.savefig(path_mej, dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ {path_mej}")
    
    # 5. R² por banco
    print("► Generando R² por banco...")
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE, facecolor='white')
    grafico_r2_por_banco(df_res, ax)
    path_r2 = OUTPUT_DIR / "r2_por_banco.png"
    fig.savefig(path_r2, dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ {path_r2}")
    
    # 6. Scatter predicción vs real
    print("► Generando scatter predicción vs real...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, facecolor='white')
    grafico_scatter_prediccion(df_proc, ax_ml=ax1, ax_vul=ax2)
    path_scatter = OUTPUT_DIR / "scatter_prediccion_vs_real.png"
    fig.savefig(path_scatter, dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ {path_scatter}")
    
    print(f"\n{'='*60}")
    print(f"✅ {5} gráficos generados en: {OUTPUT_DIR}/")
    print(f"{'='*60}")
    print(f"\n★ Imagen principal para portafolio:")
    print(f"  → {path_resumen}")
    print(f"\n→ Siguiente paso: README + GitHub")
