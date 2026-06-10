
# CONFIGURACION


FOLDER_TSV    = r"\\imgserver\IMAGES\CONFOCAL\IA\crodriguezj\data\1700_microns"
OUTPUT_FOLDER = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\1700MICRAS\CENTROIDE\FEATURE_ENGINEERING\new_eda2"

N_BINS = 10

# IMPORTS

import os, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
def out(f): return os.path.join(OUTPUT_FOLDER, f)


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
    # Motivo: redundantes con Mean+Std; muy sensibles a outliers

    l1_cols = [c for c in df_num.columns
               if 'Min Intensity' in c or 'Max Intensity' in c]
    df_num = df_num.drop(columns=l1_cols)
    if len(lista_dfs) == 0:
        print(f"  [L1] {len(l1_cols)} columnas eliminadas (Min/Max Intensity)")

    #  L2: Eliminar Entropy 32bins (9 cols), conservar 64bins
    # Motivo: correlación r > 0.994 con Entropy 64bins en todos

    l2_cols = [c for c in df_num.columns if 'Entropy 32bins' in c]
    df_num = df_num.drop(columns=l2_cols)
    if len(lista_dfs) == 0:
        print(f"  [L2] {len(l2_cols)} columnas eliminadas (Entropy 32bins)")
    
    # L2b: Eliminar Std Intensity — r > 0.85 con Entropy 64bins
    l2b_cols = [c for c in df_num.columns if 'Std Intensity' in c]
    df_num = df_num.drop(columns=l2b_cols)
    if len(lista_dfs) == 0:
        print(f"  [L2b] {len(l2b_cols)} columnas eliminadas (Std Intensity)")

    #  L3: Eliminar interfaces simétricas (3 cols)

    # miden lo mismo (r > 0.999). Se conserva la dirección
    # canónica A→B según orden alfabético de las clases.

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

    corr = df_num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    df_red = df_num.drop(columns=to_drop)
    print(f"  Variables eliminadas por correlacion: {len(to_drop)}")

    df_final = df_red.copy()

    # A. Areas → log + proporcion (por imagen)
    area_cols = [c for c in df_red.columns if 'Area (um2)' in c]
    for col in area_cols:
        df_final[col] = np.log1p(df_red[col])
    total_log = df_final[area_cols].sum(axis=1)
    for col in area_cols:
        df_final[col] = df_final[col] / total_log.replace(0, 1)

    # B. Entropia → log2(64) — ahora usamos 64bins
    for col in [c for c in df_red.columns if 'Entropy 64bins' in c]:
        df_final[col] = (df_red[col] / np.log2(64)).clip(0, 1)

    # C. Morfologia → clipping extremos p1-p99 (por imagen)
    for col in [c for c in df_red.columns if any(x in c for x in
                ['Solidity', 'Eccentricity', 'Form Factor', 'Convexity'])]:
        q1, q99 = df_red[col].quantile(0.01), df_red[col].quantile(0.99)
        df_final[col] = df_red[col].clip(q1, q99)

    # E. Variables compuestas: Area / Morfologia (mismo compartimento)
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
            print(f"  [E] {nombre}")

    # F. Variables adicionales

    # Ratio area tumor / area no tumor
    area_t = df_red[[c for c in df_red.columns
                     if 'Area (um2)' in c and '(Tumor)' in c]].sum(axis=1)
    area_s = df_red[[c for c in df_red.columns
                     if 'Area (um2)' in c and '(No tumor/No muscle)' in c]].sum(axis=1)
    if area_t.sum() > 0 and area_s.sum() > 0:
        df_final['Ratio_Area_Tumor_vs_Stroma'] = area_t / (area_s + EPS)
        print(f"  [F] Ratio_Area_Tumor_vs_Stroma")

    # Heterogeneidad morfologica tumor: Eccentricity - Solidity
    for e_col in [c for c in df_red.columns if 'Eccentricity' in c and '(Tumor)' in c]:
        for s_col in [c for c in df_red.columns if 'Solidity' in c and '(Tumor)' in c]:
            df_final['Ecc_minus_Solidity_(Tumor)'] = df_red[e_col] - df_red[s_col]
            print(f"  [F] Ecc_minus_Solidity_(Tumor)")

    # Carga tumoral ponderada por irregularidad: Area * Eccentricity
    for a_col in [c for c in df_red.columns if 'Area (um2)' in c and '(Tumor)' in c]:
        for e_col in [c for c in df_red.columns if 'Eccentricity' in c and '(Tumor)' in c]:
            df_final['Area_x_Eccentricity_(Tumor)'] = df_red[a_col] * df_red[e_col]
            print(f"  [F] Area_x_Eccentricity_(Tumor)")

    # D. Intensidades e Interfaces se normalizaran globalmente despues del concat

    # 
    # Z: Z-score por imagen

    # cols_num = df_final.select_dtypes(include=[np.number]).columns.tolist()
    # mu_img   = df_final[cols_num].mean()
    # sd_img   = df_final[cols_num].std().replace(0, 1)
    # df_final[cols_num] = (df_final[cols_num] - mu_img) / sd_img

    df_final['image_id'] = file
    df_final['Case'] = df['Case']
    for col in cols_presentes:
        df_final[col] = df[col]

    lista_dfs.append(df_final)


# 2. CONCAT + NORMALIZACION GLOBAL

df_all = pd.concat(lista_dfs, ignore_index=True)
df_all = df_all.reindex(sorted(df_all.columns), axis=1)
df_all = df_all.replace([np.inf, -np.inf], np.nan)
df_all = df_all.fillna(df_all.median(numeric_only=True))
print(f"\nShape global: {df_all.shape}")

# D. Intensidades normalizacion global [p1,p99]
int_cols = [c for c in df_all.select_dtypes(include=[np.number]).columns if 'Intensity' in c]
for col in int_cols:
    q1, q99 = df_all[col].quantile(0.01), df_all[col].quantile(0.99)
    df_all[col] = ((df_all[col] - q1) / (q99 - q1 + 1e-12)).clip(0, 1)
print(f"Intensidades normalizadas globalmente: {len(int_cols)}")

# D. Interfaces  normalizacion global [p1,p99] 
ifc_cols = [c for c in df_all.select_dtypes(include=[np.number]).columns
            if 'Interface' in c or 'Connectivity' in c]
for col in ifc_cols:
    q1, q99 = df_all[col].quantile(0.01), df_all[col].quantile(0.99)
    df_all[col] = ((df_all[col] - q1) / (q99 - q1 + 1e-12)).clip(0, 1)
print(f"Interfaces normalizadas globalmente: {len(ifc_cols)}")

# E+F. Variables compuestas normalizacion global 
vars_compuestas = [c for c in df_all.select_dtypes(include=[np.number]).columns
                   if any(x in c for x in ['Area_div_', 'Ratio_Area_',
                                            'Ecc_minus_', 'Area_x_'])]
for col in vars_compuestas:
    q1, q99 = df_all[col].quantile(0.01), df_all[col].quantile(0.99)
    df_all[col] = ((df_all[col] - q1) / (q99 - q1 + 1e-12)).clip(0, 1)
print(f"Variables compuestas normalizadas globalmente: {len(vars_compuestas)}")

print("Normalizacion global completada.")

# 3. MATRIZ DE CORRELACION

cols_bio = [c for c in df_all.columns if any(
    x in c for x in ['(Tumor)', '(Muscle)', '(No tumor/No muscle)', 'Interface',
                      'Area_div_', 'Ratio_Area_', 'Ecc_minus_', 'Area_x_'])]
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
ax.set_title('Matriz de Correlacion Global (1700 um)', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(out("01_correlacion_global.png"), dpi=200, bbox_inches='tight')
plt.close()
print("Correlacion guardada.")


# 4. PREPARAR DATOS PARA PCA

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
ax.set_title(f'PCA 2 componentes — varianza: {pca2.explained_variance_ratio_.sum():.1%}')
ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]:.1%})')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out("02_PCA_2componentes.png"), dpi=200, bbox_inches='tight')
plt.close()
print(f"PCA 2C: varianza = {pca2.explained_variance_ratio_.sum():.2%}")

#  6. SCREE PLOT + PCA 20 COMPONENTES

pca_full = PCA()
pca_full.fit(X_scaled)
ev = pca_full.explained_variance_ratio_
cum_ev = np.cumsum(ev)
n_pcs = len(ev)

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(range(1, n_pcs+1), ev*100, color='#4C72B0', alpha=0.75, label='Individual (%)')
ax.plot(range(1, n_pcs+1), cum_ev*100, color='#DD8452', lw=2, marker='o', ms=4,
        label='Acumulada (%)')
ax.axvline(x=2,  color='gray',    ls=':', lw=1.5, label=f'PC=2  ({cum_ev[1]:.1%})')
ax.axvline(x=20, color='crimson', ls='--', lw=2,
           label=f'PC=20 ({cum_ev[min(19,n_pcs-1)]:.1%})')
ax.set_xlabel('Numero de PCs', fontsize=12)
ax.set_ylabel('Varianza explicada (%)', fontsize=12)
ax.set_title('Scree Plot PCA — 1700 um', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', ls=':', alpha=0.5)
plt.tight_layout()
plt.savefig(out("03_scree_plot.png"), dpi=200, bbox_inches='tight')
plt.close()

n_comp = min(20, df_clean.shape[1])
pca20 = PCA(n_components=n_comp)
pc20_coords = pca20.fit_transform(X_scaled)
for i in range(n_comp):
    df_all[f'PC{i+1}'] = pc20_coords[:, i]
print(f"PCA 20C: varianza = {pca20.explained_variance_ratio_.sum():.2%}")

# 7. LOADINGS PONDERADOS

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
ax.set_title('Top 20 variables — importancia ponderada (20 PCs)', fontsize=12, fontweight='bold')
ax.set_xlabel('sqrt(Sum(loading^2 * varianza_explicada_k))')
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
    ax.set_title(f'Top {N_SHOW} — {pc_name} '
                 f'({pca20.explained_variance_ratio_[int(pc_name[2:])-1]:.1%} varianza)',
                 fontsize=12, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
plt.suptitle('Loadings PC1 y PC2', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(out("05_loadings_PC1_PC2.png"), dpi=200, bbox_inches='tight')
plt.close()
print("Loadings guardados.")
# 8. BINNING 10x10  REPRESENTATIVO  CENTROIDE DEL BIN

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

reps = []
for i in range(N_BINS):
    for j in range(N_BINS):
        cx = (pc1_edges[i] + pc1_edges[i+1]) / 2
        cy = (pc2_edges[j] + pc2_edges[j+1]) / 2
        mask = (df_all['PC1_bin'] == i) & (df_all['PC2_bin'] == j)
        sub = df_all[mask]
        if sub.empty:
            continue
        dists = cdist(sub[['PC1', 'PC2']].values, [[cx, cy]])
        idx_closest = sub.index[dists.argmin()]
        rep = df_all.loc[idx_closest].copy()
        rep['bin_id']           = f"bin_{i}_{j}"
        rep['bin_col']          = i
        rep['bin_row']          = j
        rep['num_tiles_in_bin'] = len(sub)
        rep['bin_X1'] = pc1_edges[i];  rep['bin_X2'] = pc1_edges[i+1]
        rep['bin_Y1'] = pc2_edges[j];  rep['bin_Y2'] = pc2_edges[j+1]
        for col in COLS_GUIA_ENVS:
            rep[col] = df_all.loc[idx_closest, col] if col in df_all.columns else np.nan
        reps.append(rep)

df_reps = pd.DataFrame(reps).reset_index(drop=True)
print(f"\nBins con datos: {len(df_reps)}")


# 9. MAPA DE DENSIDAD PCA

fig, ax = plt.subplots(figsize=(14, 9))
ax.scatter(df_all['PC1'], df_all['PC2'], alpha=0.03, s=1, c='gray', label='Todos los tiles')
for x in pc1_edges: ax.axvline(x, ls='--', color='blue', alpha=0.1)
for y in pc2_edges: ax.axhline(y, ls='--', color='blue', alpha=0.1)
sc = ax.scatter(df_reps['PC1'], df_reps['PC2'],
                s=df_reps['num_tiles_in_bin'] * 0.2,
                c=df_reps['num_tiles_in_bin'], cmap='viridis',
                edgecolors='black', lw=0.8, alpha=0.9, label='Representante')
plt.colorbar(sc, ax=ax, label='No tiles en bin')
ax.set_xlabel(f"PC1 ({pca20.explained_variance_ratio_[0]:.1%})", fontsize=12)
ax.set_ylabel(f"PC2 ({pca20.explained_variance_ratio_[1]:.1%})", fontsize=12)
ax.set_title(f'Mapa de Densidad PCA 1700 um — 20 PCs\n'
             f'(PC1+PC2={pca20.explained_variance_ratio_[:2].sum():.1%} | '
             f'Total={pca20.explained_variance_ratio_.sum():.1%})',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper right', markerscale=0.5)
plt.tight_layout()
plt.savefig(out("06_mapa_densidad_PCA.png"), dpi=200, bbox_inches='tight')
plt.close()
print("Mapa de densidad guardado.")

# 10. ATLAS FENOTIPICO

vars_t = sorted([v for v in df_all.columns if '(Tumor)' in v and not v.startswith('PC')])
vars_m = sorted([v for v in df_all.columns if '(Muscle)' in v])
vars_s = sorted([v for v in df_all.columns if '(No tumor/No muscle)' in v])
n_rows = max(len(vars_t), len(vars_m), len(vars_s))

if n_rows > 0:
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows * 4))
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)
    col_titles = ['TUMOR', 'MUSCULO', 'NO TUMOR / NO MUSCLE']
    for row in range(n_rows):
        for col, grp in enumerate([vars_t, vars_m, vars_s]):
            ax = axes[row, col]
            if row < len(grp):
                var = grp[row]
                gd = df_all.groupby(['PC2_bin', 'PC1_bin'])[var].mean().unstack()
                gd = gd.reindex(index=np.arange(N_BINS-1, -1, -1),
                                 columns=np.arange(N_BINS))
                sns.heatmap(gd, ax=ax, cmap='magma', vmin=0, vmax=1,
                            cbar_kws={'label': 'Media'},
                            xticklabels=False, yticklabels=False)
                ax.set_title(var, fontsize=8, fontweight='bold')
                if row == 0:
                    ax.set_xlabel(col_titles[col], fontsize=12,
                                  fontweight='bold', labelpad=15)
                    ax.xaxis.set_label_position('top')
            else:
                ax.axis('off')
    plt.suptitle('Atlas Fenotipico — PCA Landscapes (20 PCs)',
                 fontsize=20, y=1.01, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out("07_atlas_fenotipico.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Atlas guardado.")

# 11. EXPORTAR CSV GUIA + df_all

mapeo = {
    'Case':                                 'Case',
    'Object Info (tile) - Object ID':       'ID_Tile_Visiopharm',
    'bin_id':                               'ID_Cuadrado_PCA',
    'bin_col':                              'bin_col',
    'bin_row':                              'bin_row',
    'Object Info (tile) - Envelope left':   'x1_Izquierda',
    'Object Info (tile) - Envelope top':    'y1_Arriba',
    'Object Info (tile) - Envelope right':  'x2_Derecha',
    'Object Info (tile) - Envelope bottom': 'y2_Abajo',
    'bin_X1': 'X1_bin', 'bin_X2': 'X2_bin',
    'bin_Y1': 'Y1_bin', 'bin_Y2': 'Y2_bin',
    'PC1': 'Posicion_PCA_X', 'PC2': 'Posicion_PCA_Y',
    'num_tiles_in_bin': 'Densidad_Tiles_Similares',
}
for i in range(3, n_comp+1):
    mapeo[f'PC{i}'] = f'PC{i}'

cols_validas = [c for c in mapeo if c in df_reps.columns]
df_export = df_reps[cols_validas].copy()
df_export.rename(columns=mapeo, inplace=True)
csv_guia = out("GUIA_CENTROIDE.csv")
df_export.to_csv(csv_guia, index=False, sep=';', encoding='utf-8-sig')
print(f"CSV guia guardado: {csv_guia}")

df_all.to_csv(out("df_all_completo.csv"), index=False)
print("df_all completo guardado.")



# py -3.12 recorte21_A_pca_y_guia.py