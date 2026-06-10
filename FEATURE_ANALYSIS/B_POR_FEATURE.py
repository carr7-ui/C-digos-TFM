

#  CONFIGURACION

FOLDER_TSV    = r"\\imgserver\IMAGES\CONFOCAL\IA\crodriguezj\data\1700_microns"
MLD_FOLDER    = r"\\imgserver\IMAGES\CONFOCAL\IA\crodriguezj\annotations\1700_microns"
CZI_FOLDER    = r"\\imgserver\IMAGES\CONFOCAL\IA\crodriguezj\images"
OUTPUT_FOLDER = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\1700MICRAS\FEATURES\SIMPLE"

PIXEL_SIZE_UM = 0.1723
ZOOM_CZI      = 0.15
N_BINS        = 10
CELDA_PX      = 160
LABEL_PX      = 22
BORDER_PX     = 6
OVERLAY_ALPHA = 0.35
CMAP_NAME     = 'magma'
TOP_N_GLOBAL  = 20
TOP_N_PC      = 10
TOLERANCIA_FRACCION = 0.6

# IMPORTS

import os, re, struct, io, warnings
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.colors import Normalize
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings('ignore')

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
def out(f): return os.path.join(OUTPUT_FOLDER, f)

# ============================================================
# # 1. LECTURA + LIMPIEZA + NORMALIZACION POR IMAGEN
# # ============================================================

# lista_dfs = []

# for file in os.listdir(FOLDER_TSV):
#     if not (file.endswith(".tsv") or file.endswith(".csv")):
#         continue
#     path = os.path.join(FOLDER_TSV, file)
#     print(f"Procesando: {file}")

#     df = pd.read_csv(path, sep='\t').fillna(0)
#     df.columns = [c.replace('Stroma', 'No tumor/No muscle') for c in df.columns]
#     df['Case'] = df['Name'] if 'Name' in df.columns else file

#     cols_guia = [
#         'Object Info (tile) - Object ID',
#         'Object Info (tile) - Envelope left',
#         'Object Info (tile) - Envelope top',
#         'Object Info (tile) - Envelope right',
#         'Object Info (tile) - Envelope bottom',
#     ]
#     cols_presentes = [c for c in cols_guia if c in df.columns]

#     meta_kw = ['Study', 'Name', 'Image', 'LayerData', 'Object ID']
#     meta_cols = [c for c in df.columns if any(x in c for x in meta_kw)]
#     df_num = df.drop(columns=meta_cols + cols_presentes, errors='ignore')
#     df_num = df_num.select_dtypes(include=[np.number])

#     corr = df_num.corr().abs()
#     upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
#     to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
#     df_red = df_num.drop(columns=to_drop)
#     print(f"  Variables eliminadas por correlacion: {len(to_drop)}")

#     df_final = df_red.copy()

#     area_cols = [c for c in df_red.columns if 'Area (um2)' in c]
#     for col in area_cols:
#         df_final[col] = np.log1p(df_red[col])
#     total_log = df_final[area_cols].sum(axis=1)
#     for col in area_cols:
#         df_final[col] = df_final[col] / total_log.replace(0, 1)

#     for col in [c for c in df_red.columns if 'Entropy 32bins' in c]:
#         df_final[col] = (df_red[col] / np.log2(32)).clip(0, 1)

#     for col in [c for c in df_red.columns if any(x in c for x in
#                 ['Solidity', 'Eccentricity', 'Form Factor', 'Convexity'])]:
#         q1, q99 = df_red[col].quantile(0.01), df_red[col].quantile(0.99)
#         df_final[col] = df_red[col].clip(q1, q99)

#     df_final['image_id'] = file
#     df_final['Case'] = df['Case']
#     for col in cols_presentes:
#         df_final[col] = df[col]

#     lista_dfs.append(df_final)



# 1. LECTURA + LIMPIEZA + NORMALIZACION POR IMAGEN

lista_dfs = []

for file in os.listdir(FOLDER_TSV):
    if not (file.endswith(".tsv") or file.endswith(".csv")):
        continue
    path = os.path.join(FOLDER_TSV, file)
    print(f"Procesando: {file}")

    df = pd.read_csv(path, sep='\t').fillna(0)
    df.columns = [c.replace('Stroma', 'No tumor/No muscle') for c in df.columns]
    df['Case'] = df['Name'] if 'Name' in df.columns else file

    cols_guia = [
        'Object Info (tile) - Object ID',
        'Object Info (tile) - Envelope left',
        'Object Info (tile) - Envelope top',
        'Object Info (tile) - Envelope right',
        'Object Info (tile) - Envelope bottom',
    ]
    cols_presentes = [c for c in cols_guia if c in df.columns]

    meta_kw = ['Study', 'Name', 'Image', 'LayerData', 'Object ID']
    meta_cols = [c for c in df.columns if any(x in c for x in meta_kw)]
    df_num = df.drop(columns=meta_cols + cols_presentes, errors='ignore')
    df_num = df_num.select_dtypes(include=[np.number])

   
    # L1: Eliminar Min/Max Intensity (18 cols)
    # redundantes con Mean+Std; sensibles a outliers.
    
    l1_cols = [c for c in df_num.columns
               if 'Min Intensity' in c or 'Max Intensity' in c]
    df_num = df_num.drop(columns=l1_cols)
    if len(lista_dfs) == 0:
        print(f"  [L1] {len(l1_cols)} columnas eliminadas (Min/Max Intensity)")

    #
    # L2: Eliminar Entropy 32bins (conservar 64bins)
    #  r > 0.994 con Entropy 64bins 
    l2_cols = [c for c in df_num.columns if 'Entropy 32bins' in c]
    df_num = df_num.drop(columns=l2_cols)
    if len(lista_dfs) == 0:
        print(f"  [L2] {len(l2_cols)} columnas eliminadas (Entropy 32bins)")

    # L2b: Eliminar Std Intensity (r > 0.85 con Entropy 64bins)
    l2b_cols = [c for c in df_num.columns if 'Std Intensity' in c]
    df_num = df_num.drop(columns=l2b_cols)
    if len(lista_dfs) == 0:
        print(f"  [L2b] {len(l2b_cols)} columnas eliminadas (Std Intensity)")

    # 
    # L3: Eliminar interfaces simétricas son r > 0.999.
    # Se conserva solo la dirección A->B (orden alfabético).
    
    iface_cols = [c for c in df_num.columns if 'Interface Length' in c]
    seen_pairs, l3_cols = set(), []
    for col in iface_cols:
        inner = col.split('(')[-1].rstrip(')')
        parts = inner.split('-')
        if len(parts) == 2:
            pair = tuple(sorted([p.strip() for p in parts]))
            if pair in seen_pairs:
                l3_cols.append(col)
            else:
                seen_pairs.add(pair)
    df_num = df_num.drop(columns=l3_cols)
    if len(lista_dfs) == 0:
        print(f"  [L3] {len(l3_cols)} columnas eliminadas (interfaces simétricas)")

    #
    # Correlacion intra-imagen 
    corr = df_num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    df_red = df_num.drop(columns=to_drop)
    print(f"  Variables eliminadas por correlacion: {len(to_drop)}")

    df_final = df_red.copy()

    # A. Areas  log + proporcion (por imagen)
    area_cols = [c for c in df_red.columns if 'Area (um2)' in c]
    for col in area_cols:
        df_final[col] = np.log1p(df_red[col])
    total_log = df_final[area_cols].sum(axis=1)
    for col in area_cols:
        df_final[col] = df_final[col] / total_log.replace(0, 1)

    # B. EntropIA
    for col in [c for c in df_red.columns if 'Entropy 64bins' in c]:
        df_final[col] = (df_red[col] / np.log2(64)).clip(0, 1)

    # C. Morfologia  clipping extremos p1-p99 (por imagen)
    for col in [c for c in df_red.columns if any(x in c for x in
                ['Solidity', 'Eccentricity', 'Form Factor', 'Convexity'])]:
        q1, q99 = df_red[col].quantile(0.01), df_red[col].quantile(0.99)
        df_final[col] = df_red[col].clip(q1, q99)

   
    # E. Variables compuestas: Area / Morfologia por compartimento
    
    EPS = 1e-6
    for comp in ['(Tumor)', '(Muscle)', '(No tumor/No muscle)']:
        area_cols_comp = [c for c in df_red.columns if 'Area (um2)' in c and comp in c]
        if not area_cols_comp:
            continue
        a_col = area_cols_comp[0]
        for morf in ['Solidity', 'Convexity', 'Eccentricity', 'Form Factor', 'Connectivity']:
            morf_cols_comp = [c for c in df_red.columns if morf in c and comp in c]
            if not morf_cols_comp:
                continue
            m_col = morf_cols_comp[0]
            nombre = f'Area_div_{morf.replace(" ", "_")}_{comp}'
            df_final[nombre] = df_red[a_col] / df_red[m_col].clip(lower=EPS)
            if len(lista_dfs) == 0:
                print(f"  [E] {nombre}")

    #  F. Variables adicionales (ratios e interacciones)
    
    area_t = df_red[[c for c in df_red.columns
                     if 'Area (um2)' in c and '(Tumor)' in c]].sum(axis=1)
    area_s = df_red[[c for c in df_red.columns
                     if 'Area (um2)' in c and '(No tumor/No muscle)' in c]].sum(axis=1)
    if area_t.sum() > 0 and area_s.sum() > 0:
        df_final['Ratio_Area_Tumor_vs_Stroma'] = area_t / (area_s + EPS)
        if len(lista_dfs) == 0:
            print(f"  [F] Ratio_Area_Tumor_vs_Stroma")

    for e_col in [c for c in df_red.columns if 'Eccentricity' in c and '(Tumor)' in c]:
        for s_col in [c for c in df_red.columns if 'Solidity' in c and '(Tumor)' in c]:
            df_final['Ecc_minus_Solidity_(Tumor)'] = df_red[e_col] - df_red[s_col]
            if len(lista_dfs) == 0:
                print(f"  [F] Ecc_minus_Solidity_(Tumor)")

    for a_col in [c for c in df_red.columns if 'Area (um2)' in c and '(Tumor)' in c]:
        for e_col in [c for c in df_red.columns if 'Eccentricity' in c and '(Tumor)' in c]:
            df_final['Area_x_Eccentricity_(Tumor)'] = df_red[a_col] * df_red[e_col]
            if len(lista_dfs) == 0:
                print(f"  [F] Area_x_Eccentricity_(Tumor)")

    

    df_final['image_id'] = file
    df_final['Case'] = df['Case']
    for col in cols_presentes:
        df_final[col] = df[col]

    lista_dfs.append(df_final)
#  2. CONCAT + NORMALIZACION GLOBAL

df_all = pd.concat(lista_dfs, ignore_index=True)
df_all = df_all.reindex(sorted(df_all.columns), axis=1)
df_all = df_all.replace([np.inf, -np.inf], np.nan)
df_all = df_all.fillna(df_all.median(numeric_only=True))
print(f"\nShape global: {df_all.shape}")

for col in [c for c in df_all.select_dtypes(include=[np.number]).columns if 'Intensity' in c]:
    q1, q99 = df_all[col].quantile(0.01), df_all[col].quantile(0.99)
    df_all[col] = ((df_all[col] - q1) / (q99 - q1 + 1e-12)).clip(0, 1)

for col in [c for c in df_all.select_dtypes(include=[np.number]).columns
            if 'Interface' in c or 'Connectivity' in c]:
    q1, q99 = df_all[col].quantile(0.01), df_all[col].quantile(0.99)
    df_all[col] = ((df_all[col] - q1) / (q99 - q1 + 1e-12)).clip(0, 1)

# E+F. Variables compuestas → normalizacion global [p1,p99] → [0,1]
vars_compuestas = [c for c in df_all.select_dtypes(include=[np.number]).columns
                   if any(x in c for x in ['Area_div_', 'Ratio_Area_',
                                            'Ecc_minus_', 'Area_x_'])]
for col in vars_compuestas:
    q1, q99 = df_all[col].quantile(0.01), df_all[col].quantile(0.99)
    df_all[col] = ((df_all[col] - q1) / (q99 - q1 + 1e-12)).clip(0, 1)
print(f"Variables compuestas normalizadas globalmente: {len(vars_compuestas)}")

print("Normalizacion global completada.")

#  3. MATRIZ DE CORRELACION
#
cols_bio = [c for c in df_all.columns if any(
    x in c for x in ['(Tumor)', '(Muscle)', '(No tumor/No muscle)', 'Interface'])]
df_corr = df_all[cols_bio].select_dtypes(include=[np.number])
corr_mat = df_corr.corr()

sz = max(15, len(df_corr.columns) * 0.4)
fig, ax = plt.subplots(figsize=(sz, sz * 0.7))
mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
sns.heatmap(corr_mat, mask=mask, cmap='RdBu_r', center=0, linewidths=.5,
            square=True, cbar_kws={"shrink": .5, "label": "Pearson R"},
            xticklabels=True, yticklabels=True, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
ax.set_title('Matriz de Correlacion Global (882 um)', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(out("01_correlacion_global.png"), dpi=200, bbox_inches='tight')
plt.close()


#  4. PREPARAR DATOS PARA PCA

df_clean = df_all[cols_bio].copy()
df_clean = df_clean.dropna(thresh=len(df_clean) * 0.8, axis=1)
var = df_clean.var()
df_clean = df_clean.loc[:, var > 1e-4]
df_clean = df_clean.loc[:, df_clean.var() < df_clean.var().quantile(0.99)]
df_clean = df_clean.fillna(df_clean.median())
print(f"\nVariables para PCA: {df_clean.shape[1]}")

X_scaled = StandardScaler().fit_transform(df_clean)

# 5. PCA 2 COMPONENTES

pca2 = PCA(n_components=2)
pc2_coords = pca2.fit_transform(X_scaled)
df_all['PC1_2d'] = pc2_coords[:, 0]
df_all['PC2_2d'] = pc2_coords[:, 1]

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(df_all['PC1_2d'], df_all['PC2_2d'], alpha=0.3, s=5, c='steelblue')
ax.set_title(f'PCA 2 componentes - varianza: {pca2.explained_variance_ratio_.sum():.1%}')
ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]:.1%})')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out("02_PCA_2componentes.png"), dpi=200, bbox_inches='tight')
plt.close()

#  6. SCREE PLOT + PCA 20 COMPONENTES

pca_full = PCA()
pca_full.fit(X_scaled)
ev = pca_full.explained_variance_ratio_
cum_ev = np.cumsum(ev)
n_pcs = len(ev)

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(range(1, n_pcs+1), ev*100, color='#4C72B0', alpha=0.75, label='Individual (%)')
ax.plot(range(1, n_pcs+1), cum_ev*100, color='#DD8452', lw=2, marker='o', ms=4, label='Acumulada (%)')
ax.axvline(x=2,  color='gray',    ls=':', lw=1.5, label=f'PC=2  ({cum_ev[1]:.1%})')
ax.axvline(x=20, color='crimson', ls='--', lw=2,  label=f'PC=20 ({cum_ev[min(19,n_pcs-1)]:.1%})')
ax.set_xlabel('Numero de PCs', fontsize=12)
ax.set_ylabel('Varianza explicada (%)', fontsize=12)
ax.set_title('Scree Plot PCA - 882 um', fontsize=14, fontweight='bold')
ax.legend(fontsize=10); ax.grid(axis='y', ls=':', alpha=0.5)
plt.tight_layout()
plt.savefig(out("03_scree_plot.png"), dpi=200, bbox_inches='tight')
plt.close()

n_comp = min(20, df_clean.shape[1])
pca20 = PCA(n_components=n_comp)
pc20_coords = pca20.fit_transform(X_scaled)
for i in range(n_comp):
    df_all[f'PC{i+1}'] = pc20_coords[:, i]
print(f"PCA 20C: varianza = {pca20.explained_variance_ratio_.sum():.2%}")

#  7. LOADINGS PONDERADOS
#
var_w = pca20.explained_variance_ratio_
loadings = pd.DataFrame(
    pca20.components_.T,
    index=df_clean.columns,
    columns=[f'PC{i+1}' for i in range(n_comp)]
)
lsq = (loadings**2).multiply(var_w, axis=1)
loadings['importance'] = np.sqrt(lsq.sum(axis=1))

fig, ax = plt.subplots(figsize=(10, 8))
loadings['importance'].sort_values().tail(20).plot(kind='barh', color='#0D7377', ax=ax)
ax.set_title('Top 20 variables - importancia ponderada (20 PCs)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(out("04_top20_importancia_ponderada.png"), dpi=200, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(18, 9))
N_SHOW = 15
for ax, pc_name, col_pos, col_neg in zip(
        axes, ['PC1', 'PC2'], ['#C62828', '#2E7D32'], ['#1565C0', '#6A1B9A']):
    top = loadings[pc_name].abs().sort_values(ascending=False).head(N_SHOW)
    colors = [col_pos if loadings.loc[v, pc_name] > 0 else col_neg for v in top.index]
    ax.barh(range(N_SHOW), top.values[::-1], color=colors[::-1])
    ax.set_yticks(range(N_SHOW))
    ax.set_yticklabels(top.index[::-1], fontsize=9)
    ax.set_title(f'Top {N_SHOW} - {pc_name} ({pca20.explained_variance_ratio_[int(pc_name[2:])-1]:.1%} varianza)',
                 fontsize=12, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
plt.suptitle('Loadings PC1 y PC2', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(out("05_loadings_PC1_PC2.png"), dpi=200, bbox_inches='tight')
plt.close()


top20_global = loadings['importance'].sort_values(ascending=False).head(TOP_N_GLOBAL).index.tolist()
top10_pc1    = loadings['PC1'].abs().sort_values(ascending=False).head(TOP_N_PC).index.tolist()
top10_pc2    = loadings['PC2'].abs().sort_values(ascending=False).head(TOP_N_PC).index.tolist()

# ¡ 8. BINNING 10x10 - REPRESENTATIVO POR FEATURE
#    Cada bin x feature tiene su propio tile representativo:
#    el tile cuyo valor de esa feature es mas cercano a la media
#    del bin para esa feature.
¡

df_all['PC1_bin'] = pd.cut(df_all['PC1'], bins=N_BINS, labels=False)
df_all['PC2_bin'] = pd.cut(df_all['PC2'], bins=N_BINS, labels=False)

pc1_edges = np.linspace(df_all['PC1'].min(), df_all['PC1'].max(), N_BINS + 1)
pc2_edges = np.linspace(df_all['PC2'].min(), df_all['PC2'].max(), N_BINS + 1)

COLS_GUIA_ENVS = [
    'Object Info (tile) - Object ID',
    'Object Info (tile) - Envelope left',
    'Object Info (tile) - Envelope top',
    'Object Info (tile) - Envelope right',
    'Object Info (tile) - Envelope bottom',
]

todas_features = list(set(top20_global + top10_pc1 + top10_pc2))
features_ok    = [f for f in todas_features if f in df_all.columns]


features_extra_kw = [
    'Solidity',       # Solidity tumor y muscle
    'Connectivity',   # Connectivity tumor y muscle
    'Area_div_Solidity',
    'Area_div_Connectivity',
]
#NECESARIAS PARA GRÁFICOS DE LA MEMORIA
features_extra = [
    c for c in df_all.columns
    if any(kw in c for kw in features_extra_kw)
    and any(comp in c for comp in ['(Tumor)', '(Muscle)'])
    and c not in features_ok
]
print(f"Features extra añadidas: {len(features_extra)}")
for f in features_extra:
    print(f"  {f}")

features_ok = features_ok + features_extra


print(f"\nCalculando representativos por bin x feature ({len(features_ok)} features)...")

reps_feat = []
for i in range(N_BINS):
    for j in range(N_BINS):
        mask = (df_all['PC1_bin'] == i) & (df_all['PC2_bin'] == j)
        sub  = df_all[mask]
        if sub.empty:
            continue
        for feature in features_ok:
            if feature not in sub.columns:
                continue
            media   = sub[feature].mean()
            idx_rep = (sub[feature] - media).abs().idxmin()

            rep = df_all.loc[idx_rep].copy()
            rep['bin_id']            = f"bin_{i}_{j}"
            rep['bin_col']           = i
            rep['bin_row']           = j
            rep['feature']           = feature
            rep['media_feature_bin'] = media
            rep['valor_rep']         = df_all.loc[idx_rep, feature]
            rep['num_tiles_in_bin']  = len(sub)
            rep['bin_X1'] = pc1_edges[i];  rep['bin_X2'] = pc1_edges[i+1]
            rep['bin_Y1'] = pc2_edges[j];  rep['bin_Y2'] = pc2_edges[j+1]
            for col in COLS_GUIA_ENVS:
                rep[col] = df_all.loc[idx_rep, col] if col in df_all.columns else np.nan
            reps_feat.append(rep)

df_reps_feat = pd.DataFrame(reps_feat).reset_index(drop=True)
print(f"Total representativos (bins x features): {len(df_reps_feat)}")

#  9. MAPA DE DENSIDAD PCA

fig, ax = plt.subplots(figsize=(14, 9))
ax.scatter(df_all['PC1'], df_all['PC2'], alpha=0.03, s=1, c='gray', label='Todos los tiles')
for x in pc1_edges: ax.axvline(x, ls='--', color='blue', alpha=0.1)
for y in pc2_edges: ax.axhline(y, ls='--', color='blue', alpha=0.1)
df_reps_uniq = df_reps_feat.drop_duplicates(subset=['bin_col', 'bin_row'])
sc = ax.scatter(df_reps_uniq['PC1'], df_reps_uniq['PC2'],
                s=df_reps_uniq['num_tiles_in_bin'] * 0.2,
                c=df_reps_uniq['num_tiles_in_bin'], cmap='viridis',
                edgecolors='black', lw=0.8, alpha=0.9, label='Representante')
plt.colorbar(sc, ax=ax, label='No tiles en bin')
ax.set_xlabel(f"PC1 ({pca20.explained_variance_ratio_[0]:.1%})", fontsize=12)
ax.set_ylabel(f"PC2 ({pca20.explained_variance_ratio_[1]:.1%})", fontsize=12)
ax.set_title('Mapa de Densidad PCA 882 um - 20 PCs', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', markerscale=0.5)
plt.tight_layout()
plt.savefig(out("06_mapa_densidad_PCA.png"), dpi=200, bbox_inches='tight')
plt.close()

# 10. ATLAS FENOTIPICO

vars_t = sorted([v for v in df_all.columns if '(Tumor)' in v and not v.startswith('PC')])
vars_m = sorted([v for v in df_all.columns if '(Muscle)' in v])
vars_s = sorted([v for v in df_all.columns if '(No tumor/No muscle)' in v])
n_rows = max(len(vars_t), len(vars_m), len(vars_s))

if n_rows > 0:
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows * 4))
    if n_rows == 1: axes = np.expand_dims(axes, 0)
    for row in range(n_rows):
        for col, grp in enumerate([vars_t, vars_m, vars_s]):
            ax = axes[row, col]
            if row < len(grp):
                var = grp[row]
                gd = df_all.groupby(['PC2_bin', 'PC1_bin'])[var].mean().unstack()
                gd = gd.reindex(index=np.arange(N_BINS-1, -1, -1), columns=np.arange(N_BINS))
                # sns.heatmap(gd, ax=ax, cmap='magma', cbar_kws={'label': 'Media'},
                #             xticklabels=False, yticklabels=False)
                sns.heatmap(gd, ax=ax, cmap='magma', vmin=0, vmax=1,
            cbar_kws={'label': 'Media'},
            xticklabels=False, yticklabels=False)
                ax.set_title(var, fontsize=8, fontweight='bold')
            else:
                ax.axis('off')
    plt.suptitle('Atlas Fenotipico - PCA Landscapes (20 PCs)', fontsize=20, y=1.01, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out("07_atlas_fenotipico.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Atlas guardado.")

# 11. EXPORTAR CSV GUIA POR FEATURE + df_all

mapeo_feat = {
    'Case':                                  'Case',
    'Object Info (tile) - Object ID':        'ID_Tile_Visiopharm',
    'bin_id':                                'ID_Cuadrado_PCA',
    'bin_col':                               'bin_col',
    'bin_row':                               'bin_row',
    'feature':                               'Feature',
    'media_feature_bin':                     'Media_Feature_Bin',
    'valor_rep':                             'Valor_Representante',
    'num_tiles_in_bin':                      'Densidad_Tiles_Similares',
    'Object Info (tile) - Envelope left':    'x1_Izquierda',
    'Object Info (tile) - Envelope top':     'y1_Arriba',
    'Object Info (tile) - Envelope right':   'x2_Derecha',
    'Object Info (tile) - Envelope bottom':  'y2_Abajo',
    'bin_X1': 'X1_bin', 'bin_X2': 'X2_bin',
    'bin_Y1': 'Y1_bin', 'bin_Y2': 'Y2_bin',
    'PC1': 'Posicion_PCA_X', 'PC2': 'Posicion_PCA_Y',
}
cols_v = [c for c in mapeo_feat if c in df_reps_feat.columns]
df_exp = df_reps_feat[cols_v].copy()
df_exp.rename(columns=mapeo_feat, inplace=True)
csv_guia_feat = out("GUIA_POR_FEATURE.csv")
df_exp.to_csv(csv_guia_feat, index=False, sep=';', encoding='utf-8-sig')
print(f"CSV por feature guardado: {csv_guia_feat}")

df_all.to_csv(out("df_all_completo.csv"), index=False)

# 12. MEDIAS POR BIN para overlay de color

bin_means = (
    df_all.groupby(['PC1_bin', 'PC2_bin'])[features_ok]
    .mean().reset_index()
    .rename(columns={'PC1_bin': 'bin_col', 'PC2_bin': 'bin_row'})
)
bin_means['bin_col'] = bin_means['bin_col'].astype(int)
bin_means['bin_row'] = bin_means['bin_row'].astype(int)


def safe_name(s):
    s = re.sub(r'[\\/:*?"<>|()]', '_', s)
    return re.sub(r'_+', '_', re.sub(r'\s+', '_', s)).strip('_')[:80]

def get_font(sz, bold=False):
    for fp in [f"C:/Windows/Fonts/{'arialbd' if bold else 'arial'}.ttf",
               "C:/Windows/Fonts/calibri.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(fp).exists():
            try: return ImageFont.truetype(fp, size=sz)
            except: pass
    return ImageFont.load_default()

def val_to_color(v_norm, cmap):
    r, g, b, _ = cmap(float(np.clip(v_norm, 0, 1)))
    return (int(r*255), int(g*255), int(b*255))

def apply_overlay(img, rgb, alpha):
    ov = Image.new('RGBA', img.size, (rgb[0], rgb[1], rgb[2], int(alpha*255)))
    return Image.alpha_composite(img.convert('RGBA'), ov).convert('RGB')

def draw_border(img, rgb, width):
    d = ImageDraw.Draw(img)
    w, h = img.size
    for i in range(width):
        d.rectangle([i, i, w-1-i, h-1-i], outline=rgb)
    return img

def make_colorbar(height, vmin, vmax, cmap, bar_w=25):
    dpi = 100
    fig, ax = plt.subplots(figsize=((bar_w+80)/dpi, height/dpi), dpi=dpi)
    fig.subplots_adjust(left=0.05, right=0.45, top=0.97, bottom=0.03)
    fig.patch.set_facecolor('#1e1e1e')
    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax)
    cb.ax.tick_params(labelsize=8, colors='white')
    cb.outline.set_edgecolor('white')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='#1e1e1e')
    plt.close(fig); buf.seek(0)
    return Image.open(buf).convert('RGB').resize((bar_w+80, height), Image.LANCZOS)

def build_feature_grid(tile_lk, bm, feature, cmap, n_bins=10,
                        cpx=160, lpx=22, bpx=6, alpha=0.35):
    vals = bm[feature].dropna()
    if vals.empty or vals.nunique() < 2: return None, None, None
    vmin = float(vals.quantile(0.02)); vmax = float(vals.quantile(0.98))
    if vmin == vmax: vmin, vmax = float(vals.min()), float(vals.max())

    norm   = Normalize(vmin=vmin, vmax=vmax)
    cell_h = cpx + lpx
    grid   = Image.new('RGB', (n_bins*cpx, cell_h*n_bins), (30, 30, 30))
    bm_idx = bm.set_index(['bin_col', 'bin_row'])[feature]

    for bc in range(n_bins):
        for br in range(n_bins):
            row_plot = n_bins - 1 - br
            xoff = bc*cpx; yoff = row_plot*cell_h
            key = (bc, br)
            if key not in tile_lk or key not in bm_idx.index:
                grid.paste(Image.new('RGB', (cpx, cpx), (50, 50, 50)), (xoff, yoff))
                continue
            val = bm_idx.loc[key]
            if isinstance(val, pd.Series): val = val.iloc[0]
            if pd.isna(val): continue

            v_norm  = float(norm(val))
            col_rgb = val_to_color(v_norm, cmap)
            tile    = tile_lk[key].copy().resize((cpx, cpx), Image.LANCZOS)
            tile    = apply_overlay(tile, col_rgb, alpha)
            tile    = draw_border(tile, col_rgb, bpx)
            grid.paste(tile, (xoff, yoff))

            lum     = 0.299*col_rgb[0] + 0.587*col_rgb[1] + 0.114*col_rgb[2]
            txt_col = (255, 255, 255) if lum < 128 else (0, 0, 0)
            strip   = Image.new('RGB', (cpx, lpx), col_rgb)
            d       = ImageDraw.Draw(strip)
            font    = get_font(10)
            tb      = d.textbbox((0, 0), f"{val:.3g}", font=font)
            d.text(((cpx-(tb[2]-tb[0]))//2, (lpx-(tb[3]-tb[1]))//2),
                   f"{val:.3g}", fill=txt_col, font=font)
            grid.paste(strip, (xoff, yoff+cpx))

    return grid, vmin, vmax

def assemble_grid_image(grid, feat_name, vmin, vmax, cmap, subtitle=""):
    bar   = make_colorbar(grid.height, vmin, vmax, cmap)
    comb  = Image.new('RGB', (grid.width+bar.width, grid.height), (30, 30, 30))
    comb.paste(grid, (0, 0)); comb.paste(bar, (grid.width, 0))
    th    = 50
    title = Image.new('RGB', (comb.width, th), (20, 20, 20))
    d     = ImageDraw.Draw(title)
    ft    = get_font(13, bold=True)
    tb    = d.textbbox((0, 0), feat_name, font=ft)
    d.text((max(4, (comb.width-(tb[2]-tb[0]))//2), 4), feat_name, fill=(240, 240, 240), font=ft)
    if subtitle:
        fs  = get_font(9)
        tb2 = d.textbbox((0, 0), subtitle, font=fs)
        d.text((max(4, (comb.width-(tb2[2]-tb2[0]))//2), 30), subtitle, fill=(180, 180, 200), font=fs)
    final = Image.new('RGB', (comb.width, th+comb.height), (20, 20, 20))
    final.paste(title, (0, 0)); final.paste(comb, (0, th))
    return final

# 13. RECORTE CZI

try:
    from pylibCZIrw import czi as pyczi
except ImportError:
    pyczi = None
    print("[INFO] pylibCZIrw no disponible -> crops CZI omitidos")

def get_center_um(czidoc):
    root = ET.fromstring(czidoc.raw_metadata)
    scene = root.find('.//Scene')
    if scene is not None:
        pos = scene.find('CenterPosition')
        if pos is not None and pos.text:
            p = pos.text.strip().split(',')
            if len(p) == 2:
                try: return float(p[0]), float(p[1])
                except: pass
    for el in root.iter('CenterPosition'):
        if el.text:
            p = el.text.strip().split(',')
            if len(p) == 2:
                try: return float(p[0]), float(p[1])
                except: pass
    return None, None


def recortar_tile_czi(czidoc, x1, y1, x2, y2, pxum=0.1723, zoom=0.15):
    bbox = czidoc.total_bounding_box

    # Detectar multi-escena
    multi_escena = False
    try:
        scenes = czidoc.scenes_bounding_rectangle
        if len(scenes) > 1:
            multi_escena = True
    except Exception:
        pass

    def dentro(xs, ys, w, h):
        return (xs >= bbox['X'][0] and xs + w <= bbox['X'][1] and
                ys >= bbox['Y'][0] and ys + h <= bbox['Y'][1])

    if not multi_escena:
        # Formula original con CenterPosition
        cx_um, cy_um = get_center_um(czidoc)
        if cx_um is None: raise ValueError("Sin CenterPosition")
        cx_px = (bbox['X'][0] + bbox['X'][1]) / 2
        cy_px = (bbox['Y'][0] + bbox['Y'][1]) / 2

        def to_px(xm, ym):
            return (int(round((xm*1000 - cx_um)/pxum + cx_px)),
                    int(round(-(ym*1000 - cy_um)/pxum + cy_px)))

        px_l, py_t = to_px(x1, max(y1, y2))
        px_r, py_b = to_px(x2, min(y1, y2))
        xs = min(px_l,px_r); ys = min(py_t,py_b)
        w  = abs(px_r-px_l); h  = abs(py_b-py_t)

        if not dentro(xs, ys, w, h):
            cx2 = (bbox['X'][0]+bbox['X'][1])/2 * pxum
            cy2 = (bbox['Y'][0]+bbox['Y'][1])/2 * pxum
            px_l, py_t = (int(round((x1*1000-cx2)/pxum+(bbox['X'][0]+bbox['X'][1])/2)),
                          int(round(-(max(y1,y2)*1000-cy2)/pxum+(bbox['Y'][0]+bbox['Y'][1])/2)))
            px_r, py_b = (int(round((x2*1000-cx2)/pxum+(bbox['X'][0]+bbox['X'][1])/2)),
                          int(round(-(min(y1,y2)*1000-cy2)/pxum+(bbox['Y'][0]+bbox['Y'][1])/2)))
            xs = min(px_l,px_r); ys = min(py_t,py_b)
            w  = abs(px_r-px_l); h  = abs(py_b-py_t)
            if not dentro(xs, ys, w, h): return None

    else:
        # Intento 1: Y invertido respecto al borde inferior del bbox total
        total_y_bottom = bbox['Y'][1]
        xs = min(int(round(x1*1000/pxum)), int(round(x2*1000/pxum)))
        xe = max(int(round(x1*1000/pxum)), int(round(x2*1000/pxum)))
        yt = min(int(round(-(y1*1000/pxum)+total_y_bottom)),
                 int(round(-(y2*1000/pxum)+total_y_bottom)))
        yb = max(int(round(-(y1*1000/pxum)+total_y_bottom)),
                 int(round(-(y2*1000/pxum)+total_y_bottom)))
        w = xe - xs; h = yb - yt; ys = yt

        if not dentro(xs, ys, w, h):
            # Intento 2: CenterPosition
            cx_um, cy_um = get_center_um(czidoc)
            if cx_um is not None:
                cx_px = (bbox['X'][0]+bbox['X'][1])/2
                cy_px = (bbox['Y'][0]+bbox['Y'][1])/2
                px_l = int(round((x1*1000-cx_um)/pxum+cx_px))
                px_r = int(round((x2*1000-cx_um)/pxum+cx_px))
                py_t = int(round(-(max(y1,y2)*1000-cy_um)/pxum+cy_px))
                py_b = int(round(-(min(y1,y2)*1000-cy_um)/pxum+cy_px))
                xs = min(px_l,px_r); ys = min(py_t,py_b)
                w  = abs(px_r-px_l); h  = abs(py_b-py_t)

            if not dentro(xs, ys, w, h):
                # Intento 3: conversion directa sin invertir Y
                xs = min(int(round(x1*1000/pxum)), int(round(x2*1000/pxum)))
                ys = min(int(round(y1*1000/pxum)), int(round(y2*1000/pxum)))
                w  = abs(int(round(x2*1000/pxum)) - int(round(x1*1000/pxum)))
                h  = abs(int(round(y2*1000/pxum)) - int(round(y1*1000/pxum)))
                if not dentro(xs, ys, w, h):
                    return None

    reg = czidoc.read(roi=(xs, ys, w, h), zoom=zoom)
    if reg is None or reg.size == 0 or reg.max() == 0: return None
    img = Image.fromarray(reg[..., ::-1].astype(np.uint8))
    return img.resize((int(round(w*zoom)), int(round(h*zoom))), Image.LANCZOS)
#  14. CROPS Y GRIDS — UNO POR FEATURE
# si el mismo tile fisico aparece
#     en varias features, se recorta UNA sola vez del CZI y se
#     reutiliza la imagen en memoria. 

tiles_feat_dir = Path(OUTPUT_FOLDER) / "tiles_png_feat"
tiles_feat_dir.mkdir(exist_ok=True)

guia_feat_df = pd.read_csv(csv_guia_feat, sep=';', encoding='utf-8-sig')
guia_feat_df = guia_feat_df.rename(columns={
    'x1_Izquierda': 'env_x1', 'y1_Arriba': 'env_y1',
    'x2_Derecha':   'env_x2', 'y2_Abajo':  'env_y2',
})

# Cache en memoria: (case, x1_round, y1_round) -> PIL image o None
crop_cache = {}
tiles_ok  = 0
tiles_err = 0
czi_handles = {}   # 


def get_crop(case, row):
    """Recorta del CZI con cache. Abre el CZI solo una vez por caso."""
    global tiles_ok, tiles_err
    key = (case, round(float(row['env_x1']), 6), round(float(row['env_y1']), 6))
    if key in crop_cache:
        return crop_cache[key]

    czi_path = Path(CZI_FOLDER) / f"{case}.czi"
    if not czi_path.exists():
        crop_cache[key] = None; return None

    try:
        with pyczi.open_czi(str(czi_path)) as czidoc:
            img = recortar_tile_czi(
                czidoc,
                float(row['env_x1']), float(row['env_y1']),
                float(row['env_x2']), float(row['env_y2']),
                pxum=PIXEL_SIZE_UM, zoom=ZOOM_CZI
            )
    except Exception as e:
        print(f"    ERROR CZI {case}: {e}"); img = None

    if img is None: tiles_err += 1
    else:           tiles_ok  += 1

    crop_cache[key] = img
    return img


cmap_obj = matplotlib.colormaps[CMAP_NAME]


def procesar_feature_list(feature_list, prefix, subtitle_fn, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    grids_ok = 0; grids_skip = 0

    for rank, feature in enumerate(feature_list, 1):
        df_feat = guia_feat_df[guia_feat_df['Feature'] == feature]
        if df_feat.empty or feature not in bin_means.columns:
            grids_skip += 1; continue

        feat_safe = safe_name(feature)

        # Carpeta especifica para esta feature
        feat_crop_dir = tiles_feat_dir / feat_safe
        feat_crop_dir.mkdir(exist_ok=True)

        # Lookup de thumbnails ESPECIFICOS para esta feature
        # Cada posicion (bin_col, bin_row) tiene el tile elegido
        # como representativo de ESTA feature (puede diferir entre features)
        feat_tile_lk = {}

        if pyczi is not None:
            for _, row in df_feat.iterrows():
                bc   = int(row['bin_col'])
                br   = int(row['bin_row'])
                case = row['Case']

                # Nombre del archivo incluye el caso de origen
                png_path = feat_crop_dir / f"bin_{bc}_{br}_{case}.png"

                if png_path.exists():
                    if (bc, br) not in feat_tile_lk:
                        feat_tile_lk[(bc, br)] = Image.open(png_path)
                    continue

                img = get_crop(case, row)

                if img is not None:
                    img.save(png_path)
                    feat_tile_lk[(bc, br)] = img
                    print(f"  [{feat_safe}] bin_{bc}_{br} ({case}) OK")
                else:
                    print(f"  [{feat_safe}] bin_{bc}_{br} ({case}) VACIO")

        if not feat_tile_lk:
            grids_skip += 1; continue

        grid, vmin, vmax = build_feature_grid(
            feat_tile_lk, bin_means, feature, cmap_obj,
            n_bins=N_BINS, cpx=CELDA_PX, lpx=LABEL_PX,
            bpx=BORDER_PX, alpha=OVERLAY_ALPHA
        )
        if grid is None: grids_skip += 1; continue

        subtitle = subtitle_fn(feature, rank)
        final    = assemble_grid_image(grid, feature, vmin, vmax, cmap_obj, subtitle)
        fname    = f"{prefix}{rank:02d}_{feat_safe}.png"
        final.save(output_dir / fname, dpi=(150, 150))
        grids_ok += 1
        print(f"  [GRID OK] {fname}")

    return grids_ok, grids_skip

print("\n=== Grids features biologicas adicionales ===")
g4, s4 = procesar_feature_list(
    features_extra, "BIO_",
    lambda f, r: f"Feature biologica adicional {r}/{len(features_extra)}",
    Path(out("grids_bio_extra"))
)
print(f"  Grids OK: {g4} | Saltados: {s4}")


print("\n=== Grids Top20 (importancia ponderada) ===")
g1, s1 = procesar_feature_list(
    top20_global, "TOP",
    lambda f, r: f"Top {r}/{TOP_N_GLOBAL} importancia ponderada - 20 PCs",
    Path(out("grids_top20"))
)

print("\n=== Grids Top10 PC1 ===")
g2, s2 = procesar_feature_list(
    top10_pc1, "PC1_",
    lambda f, r: f"PC1 rank {r}/{TOP_N_PC} | loading {loadings.loc[f,'PC1']:+.3f}",
    Path(out("grids_pc1"))
)

print("\n=== Grids Top10 PC2 ===")
g3, s3 = procesar_feature_list(
    top10_pc2, "PC2_",
    lambda f, r: f"PC2 rank {r}/{TOP_N_PC} | loading {loadings.loc[f,'PC2']:+.3f}",
    Path(out("grids_pc2"))
)

#  15. SOLAPAMIENTO MLD

COLOR_CAT = {
    "tumor":  ('#E74C3C', 0.45, "Tumor"),
    "stroma": ('#3498DB', 0.45, "Estroma/Fibroso"),
    "muscle": ('#2ECC71', 0.45, "Musculo"),
    "bk":     (None,      0.0,  "Background"),
    "otros":  ('#FF00FF', 0.40, "Otros"),
}
LABEL_NUM = {1:"stroma",2:"tumor",3:"bk",4:"stroma",
             5:"muscle",6:"stroma",7:"stroma",8:"stroma",9:"stroma"}
TID_DEF   = {0:"stroma",1:"stroma",2:"tumor",3:"bk",4:"muscle",5:"muscle"}

def normalizar_cat(nombre):
    if not nombre: return None
    n = nombre.lower()
    if any(x in n for x in ['tumor','tumour']): return "tumor"
    if any(x in n for x in ['stroma','fibrous']): return "stroma"
    if any(x in n for x in ['muscle','muscul']): return "muscle"
    if any(x in n for x in ['bk','background']): return "bk"
    if any(x in n for x in ['vessel','vasc']): return "stroma"
    m = re.search(r'label\s*0*(\d+)', n)
    if m: return LABEL_NUM.get(int(m.group(1)), "otros")
    return None

def categoria_obj(obj, nl):
    tid = obj['type_id']
    if tid in nl:
        cat = normalizar_cat(nl[tid])
        if cat: return cat
    return TID_DEF.get(tid, "otros")

def leer_layer_configs(path):
    res = {}
    try:
        with open(path, "rb") as f: content = f.read()
        tag = b"[LayerConfigs]"; idx = content.find(tag)
        if idx == -1: return res
        ptr = idx + len(tag) + 1
        length = struct.unpack("<Q", content[ptr:ptr+8])[0]
        xml_s = content[ptr+8:ptr+8+length].decode("utf-8", errors="ignore")
        root = ET.fromstring(xml_s)
        for ln in root.findall("Layer"):
            lname = ln.get("Name", ""); res[lname] = {}
            for tn in ln.findall("Type"):
                ia = tn.get("Index"); nn = tn.find("n")
                if ia and nn is not None and nn.text:
                    res[lname][int(ia)] = nn.text.strip()
    except Exception as e: print(f"  [LayerConfigs] {e}")
    return res

def leer_mld(path):
    objetos = []
    with open(path, "rb") as f:
        assert f.read(4).decode("ascii","ignore") == "LDFF"
        f.read(4); n_layers = struct.unpack("<i", f.read(4))[0]
        for _ in range(n_layers):
            hdr = f.read(69)
            if len(hdr) < 69: break
            lname = hdr[:64].split(b'\x00')[0].decode("ascii","ignore").strip()
            n_obj = struct.unpack("<i", hdr[65:69])[0]; read = 0
            while read < n_obj:
                sr = f.read(4)
                if not sr: break
                buf = f.read(struct.unpack("<i", sr)[0]); off = 0
                while off < len(buf) and read < n_obj:
                    try:
                        shape = buf[off]; off+=1; ttype=buf[off]; off+=1
                        pts=[]; bbox=None
                        if shape in [0,8]:
                            np_=struct.unpack("<i",buf[off:off+4])[0]; off+=4
                            for _ in range(np_):
                                x=struct.unpack("<f",buf[off:off+4])[0]
                                y=struct.unpack("<f",buf[off+4:off+8])[0]
                                pts.append((x,y)); off+=8
                            if pts:
                                xs,ys=zip(*pts); bbox=(min(xs),min(ys),max(xs),max(ys))
                        elif shape==5:
                            off+=4
                            cx=struct.unpack("<d",buf[off:off+8])[0]; off+=8
                            cy=struct.unpack("<d",buf[off:off+8])[0]; off+=8
                            w=struct.unpack("<d",buf[off:off+8])[0]; off+=8
                            h=struct.unpack("<d",buf[off:off+8])[0]; off+=8
                            _=struct.unpack("<d",buf[off:off+8])[0]; off+=8
                            if w<1e30 and h<1e30:
                                bbox=(cx-w/2,cy-h/2,cx+w/2,cy+h/2)
                                pts=[(cx-w/2,cy-h/2),(cx+w/2,cy-h/2),(cx+w/2,cy+h/2),(cx-w/2,cy+h/2)]
                        elif shape==6:
                            off+=4
                            cx=struct.unpack("<d",buf[off:off+8])[0]; off+=8
                            cy=struct.unpack("<d",buf[off:off+8])[0]; off+=8
                            wh=struct.unpack("<d",buf[off:off+8])[0]; off+=8
                            _=struct.unpack("<d",buf[off:off+8])[0]; off+=8
                            if wh<1e30:
                                bbox=(cx-wh/2,cy-wh/2,cx+wh/2,cy+wh/2)
                                pts=[(cx-wh/2,cy-wh/2),(cx+wh/2,cy-wh/2),(cx+wh/2,cy+wh/2),(cx-wh/2,cy+wh/2)]
                        elif shape==1: off+=4+16+24
                        elif shape==2: off+=4+16+8
                        elif shape==3:
                            np2=struct.unpack("<i",buf[off:off+4])[0]; off+=4; off+=np2*8
                        elif shape in [4,9]: off+=16
                        elif shape==7: off+=20
                        for _ in range(2):
                            while off<len(buf) and buf[off]!=0: off+=1
                            off+=1
                        if bbox is not None:
                            objetos.append({'layer':lname,'type_id':ttype,'points':pts,
                                            'bbox':bbox,'w':abs(bbox[2]-bbox[0]),'h':abs(bbox[3]-bbox[1])})
                        read+=1
                    except Exception: break
    return objetos

def separar_mld(objetos, nombres_por_capa):
    roi_cand = [o for o in objetos
        if o['layer'] == 'ROI' and 0.2 < o['w'] < 2.5 and abs(o['w']-o['h']) < 0.2]
    tid=Counter(o['type_id'] for o in roi_cand).most_common(1)[0][0] if roi_cand else None
    tiles = [o for o in objetos
     if o['layer'] == 'ROI' and o['type_id'] == tid and o['w'] < 3.0]
    nl=nombres_por_capa.get('Label',{})
    anots=[]
    for o in objetos:
        if o['layer']!='Label' or o['w']>5 or o['h']>5: continue
        cat=categoria_obj(o,nl)
        if cat=='bk': continue
        o['categoria']=cat; anots.append(o)
    ts=float(np.median([t['w'] for t in tiles])) if tiles else 0.353
    print(f"  Tiles ROI: {len(tiles)} ({ts:.4f}mm) | Anotaciones: {len(anots)}")
    return tiles, anots, ts


Path(out("solapamiento_mld")).mkdir(exist_ok=True)

df_guia_unico = guia_feat_df.drop_duplicates(subset=['Case', 'env_x1', 'env_y1']).copy()

procesados = 0; errores_mld = 0
for case, grupo in df_guia_unico.groupby('Case'):
    mld_path = Path(MLD_FOLDER) / f"{case}.mld"
    if not mld_path.exists():
        print(f"[AVISO] MLD no encontrado: {case}"); errores_mld += 1; continue

    print(f"\n=== SOLAPAMIENTO {case} ({len(grupo)} tiles unicos) ===")
    nombres = leer_layer_configs(str(mld_path))
    try: objetos = leer_mld(str(mld_path))
    except Exception as e: print(f"  [ERROR] {e}"); errores_mld += 1; continue

    tiles_mld, anots, ts = separar_mld(objetos, nombres)
    if not tiles_mld: print("  Sin tiles MLD"); errores_mld += 1; continue

    centros = np.array([((t['bbox'][0]+t['bbox'][2])/2,
                         (t['bbox'][1]+t['bbox'][3])/2) for t in tiles_mld])

    indices_rep = set()
    for _, row in grupo.iterrows():
        cx = (row['env_x1']+row['env_x2'])/2
        cy = (row['env_y1']+row['env_y2'])/2
        dists = np.sqrt((centros[:,0]-cx)**2+(centros[:,1]-cy)**2)
        imin  = dists.argmin()
        if dists[imin] <= ts * TOLERANCIA_FRACCION:
            indices_rep.add(imin)
        else:
            print(f"  [sin match] {row.get('ID_Cuadrado_PCA','')} dist={dists[imin]:.4f}mm")
    print(f"  Marcados: {len(indices_rep)}/{len(grupo)}")

    fig, ax = plt.subplots(figsize=(16,12)); ax.set_aspect('equal')
    cats_usadas = set()
    for ann in sorted(anots, key=lambda o: o['w']*o['h'], reverse=True):
        cat=ann.get('categoria','otros'); c,a,_=COLOR_CAT.get(cat,COLOR_CAT['otros'])
        if c is None: continue
        cats_usadas.add(cat)
        if len(ann['points'])>=3:
            ax.add_patch(MplPolygon(ann['points'],closed=True,facecolor=c,edgecolor=c,alpha=a,lw=0.3,zorder=2))
    for t in tiles_mld:
        if len(t['points'])>=3:
            ax.add_patch(MplPolygon(t['points'],closed=True,lw=0.25,edgecolor='#888888',facecolor='none',alpha=0.5,zorder=3))
    for idx in indices_rep:
        if len(tiles_mld[idx]['points'])>=3:
            ax.add_patch(MplPolygon(tiles_mld[idx]['points'],closed=True,lw=2.0,edgecolor='#E67E00',facecolor='#FF8C00',alpha=0.8,zorder=5))
    xs1=[t['bbox'][0] for t in tiles_mld]; xs2=[t['bbox'][2] for t in tiles_mld]
    ys1=[t['bbox'][1] for t in tiles_mld]; ys2=[t['bbox'][3] for t in tiles_mld]
    m=ts*3; ax.set_xlim(min(xs1)-m,max(xs2)+m); ax.set_ylim(min(ys1)-m,max(ys2)+m)
    ley=[]
    for cat in ['tumor','stroma','muscle','otros']:
        if cat not in cats_usadas: continue
        c,a,nom=COLOR_CAT[cat]
        if c: ley.append(mpatches.Patch(facecolor=c,alpha=a,label=nom))
    ley+=[mpatches.Patch(facecolor='none',edgecolor='#888888',lw=0.8,label=f'Todos ({len(tiles_mld)})'),
          mpatches.Patch(facecolor='#FF8C00',alpha=0.8,label=f'Representativos por feature ({len(indices_rep)})')]
    ax.legend(handles=ley,loc='upper right',fontsize=9)
    ax.set_title(f"Solapamiento MLD (por feature) - {case}",fontsize=13,fontweight='bold')
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    plt.tight_layout()
    plt.savefig(out(f"solapamiento_mld/{case}_solapamiento_feat.png"),dpi=200,bbox_inches='tight')
    plt.close()
    procesados += 1; print(f"  Guardado.")


n_unicos = sum(1 for v in crop_cache.values() if v is not None)


#py -3.12 c:/Users/carme/OneDrive/Escritorio/M.UCM/TFM/1700MICRAS/FEATURES/B_POR_FEATURE.py