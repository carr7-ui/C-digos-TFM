# Unsupervised Pattern Analysis of Bladder Cancer Invasion

TFM · Universidad Complutense de Madrid  
Análisis no supervisado de patrones de invasión en cáncer de vejiga a partir de imágenes de microscopía confocal (CZI) y anotaciones de segmentación (MLD / GeoJSON).

---

## Estructura del repositorio

```
.
├── README.md
│
├── 1_validation/
│   ├── validacion_con_tile.py
│   ├── validacion_sin_tile.py
│   └── BOXPLOTS.py
│
├── 2_quality_control/
│   └── QC_PLOTS.py
│
├── 3_feature_analysis/
│   ├── A1_EDA.py
│   ├── A2_EDA.py
│   ├── A3_EDA.py
│   └── B_POR_FEATURE.py
│
├── 4_tucker/
│   ├── tucker_grid_search.py
│   └── tucker_solapamiento_fondo_eda2.py
│
└── 5_clustering/
    └── clustering_features_eda2.py
```

---

## Flujo de ejecución

```
validacion_con_tile.py  ]
validacion_sin_tile.py  ]-->  BOXPLOTS.py  -->  QC_PLOTS.py
                        ]
                              |
                         A1_EDA.py  -->  A2_EDA.py  -->  A3_EDA.py
                         (alternativo: B_POR_FEATURE.py)
                              |
                    tucker_grid_search.py  -->  tucker_solapamiento_fondo_eda2.py
                              |
                    clustering_features_eda2.py
```

---

## Descripción de los scripts

### Fase 1 — Validación del modelo

#### validacion_con_tile.py
Valida las anotaciones del modelo de IA contra las del patólogo en imágenes con tile (múltiples escenas). Lee archivos binarios `.mld` (predicciones del modelo) y `.geojson` (anotaciones del patólogo), alinea ambos espacios de coordenadas y calcula métricas de superposición geométrica (Dice, Recall, Precision) a nivel de slide y a nivel local (ROI). Guarda los resultados en `metricas_globales.csv`.

Entrada: `.mld` + `.geojson`  
Salida: `metricas_globales.csv`, figuras de solapamiento espacial

#### validacion_sin_tile.py
Misma lógica que `validacion_con_tile.py`, adaptada a imágenes sin tile (escena única). Acumula las métricas en el mismo CSV para análisis conjunto posterior.

Entrada: `.mld` + `.geojson`  
Salida: append a `metricas_globales.csv`, figuras de solapamiento

#### BOXPLOTS.py
Lee el CSV acumulado de métricas y genera boxplots con puntos individuales mostrando la distribución de Dice y Recall por clase (Tumor, Músculo) a nivel de slide.

Entrada: `metricas_globales.csv`  
Salida: figuras `.png`

---

### Fase 2 — Quality Control

#### QC_PLOTS.py
Genera cuatro gráficas de control de calidad a partir del archivo de métricas por imagen (`image_based_metrics_objects.tsv`): cobertura de área analizada, composición tisular (tumor / músculo / otro) e intensidades medias por canal (CK, Magenta, Hematoxilina) para todos los pacientes de la cohorte.

Entrada: `image_based_metrics_objects.tsv`  
Salida: `cobertura_area_analizada.png`, `composicion_tisular.png`, `intensidades_medias.png`, `intensidades.png`

---

### Fase 3 — Análisis de features

Los scripts A1, A2 y A3 deben ejecutarse en ese orden. Cada uno lee la salida del anterior.

#### A1_EDA.py (ejecutar primero)
Lee los archivos `.tsv` de features por tile de todos los casos, aplica limpieza (eliminación de columnas Min/Max Intensity, Entropy 32bins, Std Intensity), normalización por imagen y PCA. Selecciona para cada bin del espacio PCA el tile más cercano al centroide de ese bin. Genera `df_all_completo.csv` y `GUIA_CENTROIDE.csv`.

Entrada: carpeta de `.tsv` por caso  
Salida: `df_all_completo.csv`, `GUIA_CENTROIDE.csv`, figuras PCA

#### A2_EDA.py (ejecutar segundo)
Lee `GUIA_CENTROIDE.csv` y los archivos `.mld` para verificar visualmente el solapamiento entre los tiles representativos seleccionados por PCA y las anotaciones histológicas (tumor, estroma, músculo). Genera una figura de solapamiento por caso.

Entrada: `GUIA_CENTROIDE.csv`, archivos `.mld`  
Salida: `solapamiento_mld/<caso>_solapamiento.png` por cada caso

#### A3_EDA.py (ejecutar tercero)
Lee `GUIA_CENTROIDE.csv` y para cada tile representativo abre el archivo `.czi` correspondiente y recorta la región de imagen real. Filtra recortes vacíos o inválidos. Ensambla un grid 10x10 de thumbnails en el espacio PCA.

Entrada: `GUIA_CENTROIDE.csv`, archivos `.czi`  
Salida: `tiles_png/<caso>/`, `08_grid_thumbnails_PCA.png`

#### B_POR_FEATURE.py (alternativo a A1)
Variante de selección de tiles representativos: en lugar del centroide global del bin PCA, selecciona el tile cuyo vector de features originales es más cercano a la media de cada feature dentro del bin. Permite comparar la selección por centroide PCA con la selección por media de features originales.

Entrada: carpeta de `.tsv` por caso, archivos `.mld`, `.czi`  
Salida: grid de thumbnails, figuras de solapamiento

---

### Fase 4 — Descomposición Tucker

#### tucker_grid_search.py (ejecutar primero)
Construye un tensor 3D (casos x bins_PC1 x bins_PC2) a partir de `df_all_completo.csv` y realiza un grid search sobre el número de patrones de paciente (`rp`) y patrones espaciales (`rs`) en la descomposición Tucker no-negativa (librería `tensorly`). Evalúa el error de reconstrucción con múltiples seeds para estabilidad. Genera curvas de codo y heatmap de error.

Entrada: `df_all_completo.csv`  
Salida: `GRID_SEARCH_resultados.csv`, `GRID_SEARCH_curva_error.png`, `GRID_SEARCH_heatmap_error.png`

#### tucker_solapamiento_fondo_eda2.py (ejecutar segundo)
Aplica la descomposición Tucker con el número óptimo de patrones determinado en el grid search. Visualiza los patrones espaciales en el espacio PCA, superpone las anotaciones histológicas MLD, recorta thumbnails CZI representativos de cada patrón y genera mosaicos.

Entrada: `df_all_completo.csv`, `GUIA_CENTROIDE.csv`, archivos `.mld`, `.czi`  
Salida: figuras de patrones Tucker, mosaicos de crops por patrón

---

### Fase 5 — Clustering

#### clustering_features_eda2.py
Agrupa los casos (pacientes) según la composición de sus patrones Tucker. Explora distintos valores de K mediante métricas de validación interna, aplica clustering sobre la matriz de pesos Tucker por paciente y visualiza los clusters en el espacio PCA con crops representativos.

Entrada: resultados de la descomposición Tucker, `df_all_completo.csv`  
Salida: figuras de clustering por caso, CSV de asignación de cluster por paciente

---

## Requisitos

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
pip install tensorly
pip install shapely
pip install Pillow
pip install pylibCZIrw
```

Los scripts A3_EDA.py y tucker_solapamiento_fondo_eda2.py requieren Python 3.12 por la dependencia de `pylibCZIrw`:

```bash
py -3.12 A3_EDA.py
py -3.12 tucker_solapamiento_fondo_eda2.py
```

---

## Datos de entrada

| Archivo / Carpeta | Descripción |
|---|---|
| `*.tsv` (por caso) | Features por tile extraídas del software de análisis de imagen |
| `*.mld` | Anotaciones binarias del modelo de segmentación |
| `*.geojson` | Anotaciones del patólogo (ground truth) |
| `*.czi` | Imágenes de microscopía confocal de campo completo |
| `image_based_metrics_objects.tsv` | Métricas de área e intensidad por paciente |

---

## Outputs principales

| Archivo | Generado por | Descripción |
|---|---|---|
| `metricas_globales.csv` | validacion_*.py | Dice, Recall, Precision por imagen y clase |
| `df_all_completo.csv` | A1_EDA.py | Todos los tiles con coordenadas PCA |
| `GUIA_CENTROIDE.csv` | A1_EDA.py | Tiles representativos por bin PCA |
| `GRID_SEARCH_resultados.csv` | tucker_grid_search.py | Errores Tucker por combinación rp x rs |
| `tiles_png/` | A3_EDA.py | Recortes CZI por caso y bin |

---

## Autora

Carmen · TFM · Universidad Complutense de Madrid · Máster en Bioinformática
