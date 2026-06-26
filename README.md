# 🔶 mineral_ai_synthetic

> **Predicción de Ley de Cobre (Cu%) en Modelos de Bloques Mineros usando Machine Learning**
> Pipeline completo con datos sintéticos calibrados — metodología walk-forward por banco

---

## ¿Qué problema resuelve?

En minería a cielo abierto, la planificación de producción depende de conocer
la **ley de cobre** de cada bloque antes de extraerlo. El método estándar de la
industria es la **geoestadística (Kriging)**, implementada en software como Vulcan,
Datamine o Leapfrog.

Este proyecto ataca directamente el **Factor F1** del ciclo de reconciliación minera:
la comparación entre el modelo de recursos (Vulcan/Kriging) y el control de ley real
(grade control). Es el factor más crítico de la cadena — donde se origina la mayor
parte del error que se propaga a F2 y F3.

```
Recursos (Vulcan)
      ↓ F1 ← mineral_ai actúa aquí
Grade Control (ley real)
      ↓ F2
Producción Minada
      ↓ F3
Planta / Producto Final
```

Este proyecto demuestra que un modelo de **Machine Learning entrenado con datos
históricos de sondajes y tronadura** supera consistentemente a la geoestadística
tradicional en F1 — sin requerir variogramas manuales ni parámetros expertos.

---

## Resultado principal

![Resumen ejecutivo](data/figuras/resumen_ejecutivo.png)

| Métrica | ML (LightGBM) | Vulcan (Kriging) |
|---|---|---|
| MAE ponderado | **0.0700%** | 0.0951% |
| R² ponderado | **0.947** | 0.911 |
| Bancos ganados | **31/35 (88.6%)** | 4/35 |
| Mejora MAE en F1 | **-26.4%** | — |

> Validación con metodología **walk-forward expanding window por banco** —
> la más rigurosa para datos mineros con estructura temporal/espacial.
> Sesgo Vulcan calibrado con **10 años de reconciliación histórica real (2004-2013)**.

---

## El ciclo de reconciliación y dónde actúa ML

La reconciliación minera compara lo planificado vs lo real a lo largo de toda
la cadena de valor. Los tres factores clave son:

| Factor | Comparación | ¿Qué mide? |
|---|---|---|
| **F1** | Modelo recursos → Grade Control | Precisión del modelo geológico |
| **F2** | Reservas → Producción minada | Efectividad del plan de minado |
| **F3** | Producción → Planta | Pérdidas en procesamiento |

**mineral_ai mejora F1** — el origen del error. Si F1 mejora, F2 y F3
se benefician automáticamente aguas abajo.

Datos históricos de operaciones reales muestran que el modelo de corto plazo
(Vulcan/Kriging) **sobreestima la ley de Cu en 8 de cada 10 años**, con un
sesgo promedio de +3.2%. Este sesgo sistemático es exactamente lo que ML
aprende a corregir.

---

## Metodología

### Walk-forward expanding window

```
Banco  5 → TRAIN: bancos 0-4  | TEST: banco 5
Banco  6 → TRAIN: bancos 0-5  | TEST: banco 6
Banco  7 → TRAIN: bancos 0-6  | TEST: banco 7
...
Banco 39 → TRAIN: bancos 0-38 | TEST: banco 39
```

- Respeta el **orden natural de explotación** (sin contaminación temporal)
- Equivalente a predecir el siguiente banco antes de extraerlo
- Simula el uso real del modelo mes a mes en operación
- Elimina el riesgo de data leakage entre train y test

### Variables del modelo (20 features)

| Categoría | Variables |
|---|---|
| Espaciales | `centroid_x`, `centroid_y`, `centroid_z`, `banco` |
| Litología | `roca_lp`, `roca_lp2`, `roca_cp` |
| Alteración | `alter`, `alt_lp`, `alt_lp2`, `alt_cp` |
| Cu soluble | `cus_lp`, `cus_mp` |
| Fierro | `fet_lp`, `fet_mp` |
| Arsénico | `as_lp`, `as_mp` |
| Densidad | `densty_lp` |
| Mediano plazo | `cut_mp`, `mot_lp` |

---

## Dataset sintético

Los datos son **100% sintéticos y de uso libre** — generados para replicar
fielmente la estructura de un yacimiento de cobre porfídico real sin exponer
información confidencial de ninguna minera.

### Características del generador

- **192,000 bloques** en grilla 3D regular (80 × 60 × 40)
- **Continuidad espacial real** — leyes simuladas con campo gaussiano
  que replica un variograma esférico (no aleatoriedad pura)
- **5.7% de cobertura** de ley real medida (fiel a operaciones reales)
- **Zona mineral** clasificada con lógica real (ratio CUS/CUT):
  gate cut≤0.2→LIX, ratio<0.20→SEC, 0.20-0.45→MIX, ≥0.45→OXI
- **Sesgo Vulcan calibrado** con 10 años de reconciliación histórica real:
  sobreestimación promedio +3.2% Cu, patrón oscilante por banco
- Litología y alteración **correlacionadas espacialmente** con la ley de Cu
- Variables de **corto y mediano plazo** (horizonte de planificación dual)

---

## Estructura del proyecto

```
mineral_ai_synthetic/
│
├── src/
│   ├── generar_dataset_sintetico.py       # Generador de datos sintéticos
│   ├── preparar_dataset_sintetico.py      # Limpieza, encoding, features
│   ├── entrenar_walk_forward_sintetico.py # Modelo + validación walk-forward
│   └── comparar_vs_baseline_sintetico.py  # Gráficos comparativos
│
├── data/
│   ├── raw/
│   │   └── mineral_sintetico_v1.csv       # Dataset generado (192K bloques)
│   ├── processed/
│   │   ├── dataset_preparado.csv          # Bloques con ley real (10,944)
│   │   ├── X.csv                          # Features
│   │   ├── y.csv                          # Target (ley Cu real)
│   │   └── metadata.json                  # Features y parámetros
│   ├── resultados/
│   │   ├── walk_forward_sintetico_detalle.csv
│   │   └── walk_forward_sintetico_resumen.json
│   └── figuras/
│       ├── resumen_ejecutivo.png
│       ├── mae_por_banco.png
│       ├── mejora_por_banco.png
│       ├── r2_por_banco.png
│       └── scatter_prediccion_vs_real.png
│
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Cómo ejecutar

### 1. Clonar y crear ambiente

```bash
git clone https://github.com/mgrandon/mineral_ai_synthetic.git
cd mineral_ai_synthetic
conda create -n mineral_ai python=3.10
conda activate mineral_ai
pip install -r requirements.txt
```

### 2. Ejecutar pipeline completo

```bash
# Paso 1: Generar datos sintéticos
python src/generar_dataset_sintetico.py

# Paso 2: Preparar dataset
python src/preparar_dataset_sintetico.py

# Paso 3: Entrenar y validar (walk-forward)
python src/entrenar_walk_forward_sintetico.py

# Paso 4: Generar gráficos comparativos
python src/comparar_vs_baseline_sintetico.py
```

### Tiempo estimado de ejecución

| Script | Tiempo aprox. |
|---|---|
| Generar datos | ~30 segundos |
| Preparar dataset | ~5 segundos |
| Walk-forward (35 bancos) | ~3-5 minutos |
| Gráficos | ~30 segundos |

---

## Requisitos

```
python >= 3.10
lightgbm >= 4.0
scikit-learn >= 1.3
pandas >= 2.0
numpy >= 1.24
scipy >= 1.11
matplotlib >= 3.7
```

---

## Contexto técnico

### ¿Por qué ML supera a Kriging en F1?

El kriging estándar asume:
- Estacionariedad (media y varianza constantes en el espacio)
- Un variograma único para todo el dominio
- Independencia entre variables geológicas

El modelo ML captura:
- **Interacciones no lineales** entre litología, alteración y ley Cu
- **Estructura vertical** (el banco domina con ~30% de importancia)
- **Transferencia entre horizontes** (mediano plazo mejora la predicción del corto)
- **Sesgo sistemático de Vulcan** — aprende a corregirlo banco a banco
- **Patrones locales** que el kriging suaviza por diseño

### Importancia de variables (modelo real validado)

| Grupo | Importancia |
|---|---|
| Litología | 26.4% |
| Cu soluble | 24.5% |
| Alteración | 23.2% |
| Espacial (banco/xyz) | 17.9% |
| Fierro + Arsénico | 5.5% |

Las tres variables primarias que cualquier geólogo reconoce (litología,
alteración, Cu soluble) dominan con **74% combinado** — el modelo tiene
sentido geológico real, no es una caja negra.

### Manejo de data leakage

Variables excluidas explícitamente por causar leakage con el target:
`umet_cp`, `umet_lp`, `umet_mp`, `mnzn`, `categ`, `categ_lp`

Estas variables son **derivadas** de la ley de cobre real y no estarían
disponibles en producción al momento de predecir.

---

## Aplicabilidad a datos reales

Este pipeline está diseñado para ser aplicado directamente a modelos de
bloques reales exportados desde Vulcan, Datamine, Leapfrog u otro software
geoestadístico, con mínimos ajustes:

1. Reemplazar `mineral_sintetico_v1.csv` por el export real del modelo de bloques
2. Ajustar nombres de columnas en `preparar_dataset_sintetico.py`
3. Verificar exclusión de variables con leakage según el dataset específico
4. Ejecutar el mismo pipeline sin cambios adicionales

**Validado en datos reales:** en un yacimiento de cobre porfídico chileno,
el modelo ML superó a Vulcan en **40/40 bancos (100%)** con una reducción
de error MAE del **21%** bajo metodología walk-forward.

---

## Autor

**Manuel Grandón Troncoso** — Consultor ML aplicado a minería
Especialización en predicción de leyes, reconciliación F1 y planificación minera

📧 [tu-email]
🔗 [LinkedIn]

---

## Licencia

MIT © 2026 Manuel Grandón Troncoso — libre uso con atribución.

---

*Este proyecto usa datos 100% sintéticos calibrados con patrones históricos
reales de reconciliación. Ningún dato confidencial de operaciones mineras
está incluido en este repositorio.*
