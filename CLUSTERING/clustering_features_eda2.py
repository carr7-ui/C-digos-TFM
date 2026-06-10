# """
# clustering_composicion_tiles.py
# ================================
# Clustering de CASOS basado en la COMPOSICION de patrones de tiles.
# OPTIMIZADO: PCA en tiles, K=4 en casos y Dominancia Relativa en Tucker.
# """

# # ============================================================
# # CONFIGURACION
# # ============================================================

# DF_ALL_CSV    = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\1700MICRAS\CENTROIDE\FEATURE_ENGINEERING\new_eda2\df_all_completo.csv"
# TUCKER_CSV    = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\1700MICRAS\CENTROIDE\FEATURE_ENGINEERING\TUCKER\CLUSTERING\CLUSTERING_resultados_por_caso.csv"
# OUTPUT_FOLDER = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\1700MICRAS\CENTROIDE\FEATURE_ENGINEERING\CLUSTERING_EDA2"

# K_TILES          = 5     # tipos de tile
# K_RANGE_CASOS    = range(2, 9)
# K_FORZADO_CASOS  = 4     # <-- MEJORA 1: Forzamos K=4 para eliminar el micro-cluster de 3 pacientes
# ZSCORE_OUTLIER   = 3.5

# CZI_FOLDER      = r"\\imgserver\IMAGES\CONFOCAL\IA\crodriguezj\images"
# PIXEL_SIZE_UM   = 0.1723
# ZOOM_CZI        = 0.15
# CROP_SIZE_PX    = (220, 220)
# MAX_CROPS_TOTAL = 50
# MAX_CROPS_CASO  = 3

# CLUSTER_COLORS = ['#E74C3C','#3498DB','#2ECC71','#F39C12','#9B59B6','#1ABC9C','#E67E22']
# TILE_COLORS    = ['#E74C3C','#3498DB','#2ECC71','#F39C12','#9B59B6']

# col_x1 = 'Object Info (tile) - Envelope left'
# col_y1 = 'Object Info (tile) - Envelope top'
# col_x2 = 'Object Info (tile) - Envelope right'
# col_y2 = 'Object Info (tile) - Envelope bottom'

# # ============================================================
# # IMPORTS
# # ============================================================

# import os, warnings
# from pathlib import Path
# from collections import Counter

# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import seaborn as sns

# from scipy import stats
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
# from sklearn.mixture import GaussianMixture
# from sklearn.decomposition import PCA as skPCA
# from sklearn.metrics import (silhouette_score, silhouette_samples,
#                              davies_bouldin_score, calinski_harabasz_score,
#                              adjusted_rand_score)
# from sklearn.utils import resample
# from scipy.cluster.hierarchy import dendrogram, linkage

# try:
#     import hdbscan
#     HDBSCAN_OK = True
# except ImportError:
#     HDBSCAN_OK = False
#     print("[AVISO] hdbscan no disponible.")

# warnings.filterwarnings('ignore')
# Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
# def out(f): return os.path.join(OUTPUT_FOLDER, f)

# # ============================================================
# # 1. CARGAR DATOS
# # ============================================================

# print("Cargando df_all_completo...")
# df = pd.read_csv(DF_ALL_CSV)
# print(f"  Shape: {df.shape}")
# print(f"  Casos: {df['Case'].nunique()}")
# print(f"  Tiles: {len(df):,}")

# N_CASOS = df['Case'].nunique()

# # ============================================================
# # 2. FEATURES BIOLOGICAS PARA CLUSTERING DE TILES
# # ============================================================

# print("\nSeleccionando features biologicas...")

# excluir = {col_x1, col_y1, col_x2, col_y2, 'Case', '_b1', '_b2',
#            '_pat_dom', '_pat_intens', '_pat_color_g'}

# cols_bio_ok = [c for c in df.columns
#                if df[c].dtype in [float,'float64']
#                if c not in excluir
#                and not c.startswith('PC')
#                and not c.startswith('_')
#                and any(x in c for x in ['(Tumor)','(Muscle)','(No tumor/No muscle)','Interface'])
#                and df[c].isna().mean() < 0.3
#                and df[c].std() > 1e-6]

# print(f"  Features válidas: {len(cols_bio_ok)}")

# X_tiles_raw = df[cols_bio_ok].copy()
# for c in cols_bio_ok:
#     X_tiles_raw[c].fillna(X_tiles_raw[c].median(), inplace=True)

# scaler_tiles = StandardScaler()
# X_tiles      = scaler_tiles.fit_transform(X_tiles_raw.values)
# print(f"  Matriz de tiles original: {X_tiles.shape}")

# # ============================================================
# # 3. CLUSTERING DE TILES (PASO 1 CON PCA INTERMEDIO)
# # ============================================================

# print(f"\n{'='*55}")
# print(f"PASO 1: CLUSTERING DE TILES (k={K_TILES}) con PCA de limpieza")
# print(f"{'='*55}")

# # --- MEJORA 2: Reducción PCA intermedia para limpiar el ruido multidimensional de los tiles ---
# print("  Aplicando reducción PCA intermedia a los tiles para maximizar nitidez fenotípica...")
# pca_reductores = skPCA(n_components=6, random_state=42) # 6 componentes limpian el ruido redundante
# X_tiles_clean = pca_reductores.fit_transform(X_tiles)

# km_tiles     = KMeans(n_clusters=K_TILES, random_state=42, n_init=30, max_iter=500)
# labels_tiles = km_tiles.fit_predict(X_tiles_clean)
# df['cluster_tile'] = labels_tiles + 1

# # La evaluación de silueta se hace sobre el espacio limpio donde clusterizamos
# sil_tiles = silhouette_score(X_tiles_clean, labels_tiles, sample_size=2000)
# print(f"  Silhouette de tiles (en espacio PCA limpio): {sil_tiles:.3f}")
# print(f"  Distribución de tiles: {dict(pd.Series(labels_tiles+1).value_counts().sort_index())}")

# # Perfil biológico (sobre variables originales para poder interpretarlo)
# df_tile_profile = pd.DataFrame(X_tiles, columns=cols_bio_ok)
# df_tile_profile['cluster_tile'] = labels_tiles + 1
# medias_tiles = df_tile_profile.groupby('cluster_tile')[cols_bio_ok].mean()
# z_tiles      = medias_tiles.apply(lambda x: (x-x.mean())/(x.std()+1e-12), axis=0)

# print("\n  Top features por cluster de tile:")
# for ct in range(1, K_TILES+1):
#     top5 = z_tiles.loc[ct].abs().nlargest(5).index.tolist()
#     vals = [(f.split('(')[0].strip()[:18], f"{z_tiles.loc[ct,f]:.2f}") for f in top5]
#     print(f"    Tile C{ct}: " + " | ".join([f"{n}={v}" for n,v in vals]))

# # Figura perfil tiles
# varianza_tiles = z_tiles.var(axis=0).sort_values(ascending=False)
# top30_tiles    = varianza_tiles.head(30).index.tolist()

# def nombre_corto(f):
#     for rep,sust in [(' (No tumor/No muscle)','\n(Str)'),(' (Muscle)','\n(Mus)'),
#                      (' (Tumor)','\n(Tum)'),('Interface Length ','Interf.'),
#                      ('Intensity','Int.'),('Entropy 32bins ','Entr.'),('Area_div_','A/')]:
#         f = f.replace(rep, sust)
#     return f[:25]

# fig, axes = plt.subplots(1, K_TILES,
#                           figsize=(4*K_TILES, max(10, len(top30_tiles)*0.35)),
#                           sharey=True, facecolor='white')
# plt.subplots_adjust(wspace=0.02)
# if K_TILES == 1: axes = [axes]
# pos_y = np.arange(len(top30_tiles))
# for ct_idx, ax in enumerate(axes):
#     ct = ct_idx+1; col = TILE_COLORS[ct_idx]
#     vals = z_tiles.loc[ct, top30_tiles].values
#     ax.barh(pos_y, vals,
#             color=['#C0392B' if v>0 else '#2980B9' for v in vals],
#             edgecolor='none', height=0.7, alpha=0.88)
#     ax.axvline(0, color='black', lw=1); ax.set_xlim(-3, 3)
#     ax.grid(axis='x', linestyle='--', alpha=0.3)
#     ax.set_title(f'Tile C{ct}\n({(labels_tiles+1==ct).sum():,} tiles)',
#                  fontsize=11, fontweight='bold', color=col)
#     ax.set_xlabel('Z-score', fontsize=8)
#     if ct == 1:
#         ax.set_yticks(pos_y)
#         ax.set_yticklabels([nombre_corto(f) for f in top30_tiles], fontsize=7)
#     else:
#         ax.tick_params(left=False)
#     for spine in ax.spines.values():
#         spine.set_edgecolor(col); spine.set_linewidth(2)
# plt.suptitle(f'Perfil Biológico de Clusters de Tiles (k={K_TILES})',
#              fontsize=13, fontweight='bold', y=0.98)
# plt.savefig(out('00_perfil_clusters_tiles.png'), dpi=180, bbox_inches='tight', facecolor='white')
# plt.close()
# print("  ✓ 00_perfil_clusters_tiles.png")

# # ============================================================
# # 4. COMPOSICION PORCENTUAL POR CASO (PASO 2)
# # ============================================================

# print(f"\n{'='*55}")
# print(f"PASO 2: COMPOSICION DE TILES POR CASO")
# print(f"{'='*55}")

# df_comp = (df.groupby('Case')['cluster_tile']
#              .value_counts(normalize=True)
#              .unstack(fill_value=0))
# for ct in range(1, K_TILES+1):
#     if ct not in df_comp.columns: df_comp[ct] = 0.0
# df_comp = df_comp[[ct for ct in range(1, K_TILES+1)]].copy()
# df_comp.columns = [f'pct_TileC{ct}' for ct in range(1, K_TILES+1)]
# df_comp = df_comp.reset_index()
# pct_cols = [f'pct_TileC{ct}' for ct in range(1, K_TILES+1)]

# print(f"  Casos: {len(df_comp)}")
# print(df_comp[pct_cols].describe().round(3).to_string())

# df_comp.to_csv(out('COMPOSICION_tiles_por_caso.csv'),
#                 index=False, sep=';', encoding='utf-8-sig', decimal=',')
# print("  ✓ COMPOSICION_tiles_por_caso.csv")

# # Heatmap composición
# fig, ax = plt.subplots(figsize=(N_CASOS*0.3+2, 5), facecolor='white')
# data_hm  = df_comp.set_index('Case')[pct_cols].T
# caso_dom = df_comp.set_index('Case')[pct_cols].idxmax(axis=1)
# orden_c  = df_comp.set_index('Case').assign(_d=caso_dom).sort_values('_d').index
# sns.heatmap(data_hm[orden_c], ax=ax, cmap='YlOrRd', vmin=0, vmax=1,
#             xticklabels=True,
#             yticklabels=[f'Tile C{ct}' for ct in range(1, K_TILES+1)],
#             cbar_kws={'label':'% tiles','shrink':0.6},
#             linewidths=0.2, linecolor='#DDDDDD')
# ax.set_xticklabels(ax.get_xticklabels(), rotation=75, ha='right', fontsize=5)
# for label, col in zip(ax.get_yticklabels(), TILE_COLORS): label.set_color(col)
# ax.set_title(f'Composición de Tipos de Tile por Caso (n={N_CASOS})',
#             fontsize=12, fontweight='bold')
# plt.tight_layout()
# plt.savefig(out('01_composicion_heatmap.png'), dpi=200, bbox_inches='tight', facecolor='white')
# plt.close()
# print("  ✓ 01_composicion_heatmap.png")

# # ============================================================
# # 5. DETECCION DE OUTLIERS
# # ============================================================

# print(f"\n{'='*55}")
# print(f"DETECCION DE OUTLIERS (z-score > {ZSCORE_OUTLIER})")
# print(f"{'='*55}")

# z_matrix  = np.abs(stats.zscore(df_comp[pct_cols].values))
# mask_ok   = (z_matrix < ZSCORE_OUTLIER).all(axis=1)
# n_outliers = (~mask_ok).sum()

# if n_outliers > 0:
#     outlier_casos = df_comp.loc[~mask_ok, 'Case'].tolist()
#     for caso in outlier_casos:
#         idx = df_comp.index[df_comp['Case']==caso][0]
#         zs  = z_matrix[idx]
#         top = sorted(zip(pct_cols, zs), key=lambda x: -x[1])[:3]
#         top_str = ', '.join([f"{c}(z={z:.1f})" for c,z in top if z>ZSCORE_OUTLIER])
#         print(f"  ⚠ OUTLIER: {caso} → {top_str}")
#     print(f"\n  Total outliers: {n_outliers}")
#     df_comp_cluster = df_comp[mask_ok].copy().reset_index(drop=True)
# else:
#     print(f"  ✓ Sin outliers")
#     outlier_casos   = []
#     df_comp_cluster = df_comp.copy()

# casos_cluster = df_comp_cluster['Case'].tolist()
# print(f"  Casos para clustering: {len(casos_cluster)}")

# scaler_casos = StandardScaler()
# X_casos      = scaler_casos.fit_transform(df_comp_cluster[pct_cols].values)

# # ============================================================
# # 6. METRICAS PARA k OPTIMO
# # ============================================================

# print(f"\n{'='*55}")
# print(f"PASO 3: CLUSTERING DE CASOS")
# print(f"{'='*55}")
# print("\nCalculando métricas para k óptimo...")

# metricas = {'k':[],'inercia':[],'silhouette':[],'davies_bouldin':[],'calinski':[]}
# for k in K_RANGE_CASOS:
#     if k >= len(casos_cluster): continue
#     km     = KMeans(n_clusters=k, random_state=42, n_init=20)
#     labels = km.fit_predict(X_casos)
#     metricas['k'].append(k)
#     metricas['inercia'].append(km.inertia_)
#     metricas['silhouette'].append(silhouette_score(X_casos, labels))
#     metricas['davies_bouldin'].append(davies_bouldin_score(X_casos, labels))
#     metricas['calinski'].append(calinski_harabasz_score(X_casos, labels))

# df_met  = pd.DataFrame(metricas)
# best_s  = int(df_met.loc[df_met['silhouette'].idxmax(), 'k'])
# best_db = int(df_met.loc[df_met['davies_bouldin'].idxmin(), 'k'])
# best_ch = int(df_met.loc[df_met['calinski'].idxmax(), 'k'])

# print(df_met.to_string(index=False))

# # Seleccionar K_FINAL dinámico o forzado
# if K_FORZADO_CASOS is not None:
#     K_FINAL = K_FORZADO_CASOS
#     print(f"\nUsando K_FINAL = {K_FINAL}  (forzado para equilibrio estructural)")
# else:
#     K_FINAL = Counter([best_s, best_db, best_ch]).most_common(1)[0][0]
#     print(f"\nUsando K_FINAL = {K_FINAL}  (votación: Sil={best_s}, DB={best_db}, CH={best_ch})")

# fig, axes = plt.subplots(2, 2, figsize=(13,9), facecolor='white')
# fig.suptitle('Métricas para k óptimo — Clustering por Composición de Tiles', fontsize=13, fontweight='bold')
# pares = [(axes[0,0],'inercia','#E74C3C','Inercia → buscar codo',None),
#          (axes[0,1],'silhouette','#3498DB','Silhouette → mayor = mejor',best_s),
#          (axes[1,0],'davies_bouldin','#2ECC71','Davies-Bouldin → menor mejor',best_db),
#          (axes[1,1],'calinski','#9B59B6','Calinski → mayor = mejor',best_ch)]
# for ax,met,col,tit,bk in pares:
#     ax.plot(df_met['k'], df_met[met], 'o-', color=col, lw=2, ms=8)
#     ax.set_title(tit, fontsize=10, fontweight='bold')
#     ax.set_xlabel('k'); ax.grid(alpha=0.3); ax.set_xticks(list(df_met['k']))
#     if bk: ax.axvline(bk, color=col, linestyle='--', alpha=0.7, label=f'mejor k={bk}')
#     if K_FINAL: ax.axvline(K_FINAL, color='black', linestyle=':', alpha=0.6, label=f'K_FINAL={K_FINAL}')
#     ax.legend(fontsize=8)
# plt.tight_layout(rect=[0, 0.05, 1, 1])
# plt.savefig(out('02_metricas_k_optimo.png'), dpi=200, bbox_inches='tight', facecolor='white')
# plt.close()

# # ============================================================
# # 7. APLICAR TODOS LOS ALGORITMOS
# # ============================================================

# print(f"\nAplicando clustering k={K_FINAL} con todos los algoritmos...")

# km_final  = KMeans(n_clusters=K_FINAL, random_state=42, n_init=50)
# labels_km = km_final.fit_predict(X_casos)

# hc_final  = AgglomerativeClustering(n_clusters=K_FINAL, linkage='ward')
# labels_hc = hc_final.fit_predict(X_casos)

# gmm_final = GaussianMixture(n_components=K_FINAL, covariance_type='full', random_state=42, n_init=20)
# gmm_final.fit(X_casos)
# labels_gm = gmm_final.predict(X_casos)
# proba_gm  = gmm_final.predict_proba(X_casos)

# spec        = SpectralClustering(n_clusters=K_FINAL, random_state=42, affinity='nearest_neighbors', n_neighbors=10)
# labels_spec = spec.fit_predict(X_casos)

# if HDBSCAN_OK:
#     clusterer   = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3, metric='euclidean')
#     labels_hdb  = clusterer.fit_predict(X_casos)
#     n_cl_hdb    = len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0)
#     n_noise_hdb = (labels_hdb == -1).sum()
# else:
#     labels_hdb, n_cl_hdb, n_noise_hdb = np.full(len(X_casos), -1), 0, len(X_casos)

# df_comp_cluster['cluster_kmeans']     = labels_km + 1
# df_comp_cluster['cluster_jerarquico'] = labels_hc + 1
# df_comp_cluster['cluster_gmm']        = labels_gm + 1
# df_comp_cluster['cluster_spectral']   = labels_spec + 1
# df_comp_cluster['cluster_hdbscan']    = labels_hdb

# # ============================================================
# # 8. DENDROGRAMA Y PROYECCIONES
# # ============================================================

# print("\nGenerando gráficos de comparación de clusters...")
# Z_link = linkage(X_casos, method='ward')
# fig, ax = plt.subplots(figsize=(max(18, len(casos_cluster)*0.28), 8), facecolor='white')
# dendrogram(Z_link, labels=list(casos_cluster), ax=ax, color_threshold=0.7*max(Z_link[:,2]), leaf_rotation=75, leaf_font_size=6)
# plt.savefig(out('03_dendrograma.png'), dpi=180, bbox_inches='tight')
# plt.close()

# pca2   = skPCA(n_components=2)
# X_2d   = pca2.fit_transform(X_casos)
# var_2d = pca2.explained_variance_ratio_

# fig, axes = plt.subplots(1, 3, figsize=(18,6), facecolor='white')
# for ax, titulo, etq in zip(axes, ['K-Means','Jerárquico (Ward)','GMM'], [labels_km+1, labels_hc+1, labels_gm+1]):
#     for k in range(1, K_FINAL+1):
#         mask = etq == k
#         ax.scatter(X_2d[mask,0], X_2d[mask,1], c=CLUSTER_COLORS[k-1], s=90, alpha=0.85, edgecolors='white', label=f'C{k}')
#         if mask.sum() > 0: ax.scatter(X_2d[mask,0].mean(), X_2d[mask,1].mean(), c=CLUSTER_COLORS[k-1], s=220, marker='*', edgecolors='black')
# plt.savefig(out('04_proyeccion_2d.png'), dpi=200, bbox_inches='tight')
# plt.close()

# # 10. Comparación 4 Algoritmos
# fig, axes = plt.subplots(1, 4, figsize=(24,6), facecolor='white')
# algoritmos = [('K-Means', labels_km+1), ('Spectral', labels_spec+1), ('Jerárquico', labels_hc+1), ('HDBSCAN', labels_hdb)]
# for ax, (titulo, etq) in zip(axes, algoritmos):
#     for k in sorted(set(etq)):
#         mask  = etq == k
#         color = '#AAAAAA' if k == -1 else CLUSTER_COLORS[(k-1) % len(CLUSTER_COLORS)]
#         ax.scatter(X_2d[mask,0], X_2d[mask,1], c=color, s=80, alpha=0.85, edgecolors='white')
# plt.savefig(out('04b_comparacion_algoritmos.png'), dpi=200, bbox_inches='tight')
# plt.close()

# # ============================================================
# # 11. PERFIL DE COMPOSICION POR CLUSTER
# # ============================================================

# medias_comp = df_comp_cluster.groupby('cluster_kmeans')[pct_cols].mean()

# fig, ax = plt.subplots(figsize=(K_FINAL*1.8+2, K_TILES*0.8+2), facecolor='white')
# im = ax.imshow(medias_comp[pct_cols].values.T*100, cmap='YlOrRd', vmin=0, vmax=100, aspect='auto')
# ax.set_xticks(range(K_FINAL)); ax.set_yticks(range(K_TILES))
# ax.set_xticklabels([f'Cluster {cl}\n(n={(labels_km+1==cl).sum()})' for cl in range(1,K_FINAL+1)], fontweight='bold')
# ax.set_yticklabels([f'Tile C{ct}' for ct in range(1,K_TILES+1)], fontweight='bold')
# for i in range(K_TILES):
#     for j in range(K_FINAL):
#         val = medias_comp[pct_cols].values.T[i,j]*100
#         ax.text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=11, fontweight='bold', color='white' if val>50 else 'black')
# plt.savefig(out('05b_composicion_heatmap.png'), dpi=200, bbox_inches='tight')
# plt.close()

# # ============================================================
# # 13. PERFIL FEATURES BIOLOGICAS POR CLUSTER
# # ============================================================

# df_feat = df[df['Case'].isin(casos_cluster)][['Case']+cols_bio_ok].merge(df_comp_cluster[['Case','cluster_kmeans']], on='Case', how='left')
# df_fc   = df_feat.groupby('cluster_kmeans')[cols_bio_ok].mean().T
# df_fc.columns = [f'C{k}' for k in df_fc.columns]
# df_fn   = df_fc.copy()
# for feat in df_fn.index:
#     row = df_fn.loc[feat]; rng = row.max()-row.min()
#     df_fn.loc[feat] = (row-row.min())/rng if rng>1e-10 else 0.5

# grupos = {'MUSCULO': [c for c in cols_bio_ok if '(Muscle)' in c], 'TUMOR': [c for c in cols_bio_ok if '(Tumor)' in c],
#           'ESTROMA': [c for c in cols_bio_ok if '(No tumor/No muscle)' in c], 'INTERFAZ': [c for c in cols_bio_ok if 'Interface' in c]}

# orden, etq, sep, pos = [], [], {}, 0
# for gnom,feats in grupos.items():
#     if not feats: continue
#     fs=sorted(feats); sep[gnom]=(pos,pos+len(fs))
#     for f in fs: orden.append(f); etq.append(f.replace(' (No tumor/No muscle)','').replace(' (Muscle)','').replace(' (Tumor)',''))
#     pos+=len(fs)

# pos_y=np.arange(len(orden)); BG=['#F0F0F0','#E8E8E8','#F0F0F0','#E8E8E8']
# fig,axes=plt.subplots(1,K_FINAL,figsize=(5*K_FINAL,max(16,len(orden)*0.22+2)), sharey=True,facecolor='white')
# plt.subplots_adjust(wspace=0.03,left=0.16,right=0.97,top=0.93,bottom=0.03)
# for k_idx,ax in enumerate(axes):
#     k=k_idx+1; col=CLUSTER_COLORS[k_idx]; cn=f'C{k}'
#     if cn not in df_fn.columns: continue
#     vals=df_fn.reindex(orden)[cn].values
#     for gi,(gnom,(g0,g1)) in enumerate(sep.items()): ax.axhspan(g0-0.5,g1-0.5,facecolor=BG[gi%4],alpha=0.5,zorder=0)
#     ax.barh(pos_y,vals,color=col,edgecolor='none',height=0.75,alpha=0.88,zorder=2)
#     if k==1:
#         ax.set_yticks(pos_y); ax.set_yticklabels(etq,fontsize=6.5)
#         for gnom,(g0,g1) in sep.items(): ax.text(-0.02,(g0+g1)/2-0.5,gnom,ha='right',va='center',fontweight='bold',transform=ax.get_yaxis_transform())
#     for spine in ax.spines.values(): spine.set_edgecolor(col); spine.set_linewidth(2.5)
# plt.savefig(out('07_perfil_features_por_cluster.png'), dpi=180, bbox_inches='tight')
# plt.close()

# # ============================================================
# # 14. COMPARACION CON PATRONES TUCKER (CON MEJORA DE Z-SCORE)
# # ============================================================

# TUCKER_SCORES_CSV  = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\1700MICRAS\CENTROIDE\FEATURE_ENGINEERING\TUCKER\new_eda2\TUCKER_contribuciones_por_caso.csv"

# if Path(TUCKER_SCORES_CSV).exists():
#     print("\n" + "="*55 + "\nCOMPARACION CON PATRONES TUCKER (MEJORADA)\n" + "="*55)
#     df_tucker_scores = pd.read_csv(TUCKER_SCORES_CSV, sep=';')
#     pat_tucker_cols  = [c for c in df_tucker_scores.columns if c.startswith('Patron_')]
#     N_PAT_T = len(pat_tucker_cols); COLORS_PAT_T = ['#2ECC71','#F39C12','#E74C3C','#9B59B6','#3498DB']

#     # --- MEJORA 3: Calcular dominancia por Z-score relativo de columna para evitar el sesgo del Patrón 4 ---
#     tucker_z = df_tucker_scores[pat_tucker_cols].apply(lambda x: (x - x.mean()) / (x.std() + 1e-12), axis=0)
#     df_tucker_scores['patron_dominante_relativo'] = tucker_z.idxmax(axis=1)

#     df_merge = df_comp_cluster[['Case','cluster_kmeans']].merge(
#         df_tucker_scores[['Case','patron_dominante_relativo'] + pat_tucker_cols], on='Case', how='inner'
#     )

#     # 1. Tabla de contingencia cruzada
#     tabla_dom = pd.crosstab(df_merge['cluster_kmeans'], df_merge['patron_dominante_relativo'], margins=True)
#     print("\n  Patrón Tucker dominante RELATIVO por cluster de composición:")
#     print(tabla_dom.to_string())
#     tabla_dom.to_csv(out('COMPARACION_cluster_vs_patron_tucker_dominante.csv'), sep=';', encoding='utf-8-sig')

#     # 2. Heatmap de contingencia
#     tabla_pct = pd.crosstab(df_merge['cluster_kmeans'], df_merge['patron_dominante_relativo'], normalize='index') * 100
#     cols_ord = sorted(tabla_pct.columns, key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
#     tabla_pct = tabla_pct.reindex(columns=cols_ord, fill_value=0)

#     fig, ax = plt.subplots(figsize=(N_PAT_T*1.5+2, K_FINAL*0.9+2), facecolor='white')
#     ax.imshow(tabla_pct.values, cmap='YlOrRd', vmin=0, vmax=100, aspect='auto')
#     ax.set_xticks(range(len(cols_ord))); ax.set_xticklabels([f"P{c.split('_')[1]}" for c in cols_ord], fontweight='bold')
#     ax.set_yticks(range(K_FINAL)); ax.set_yticklabels([f'Cluster {k}\n(n={(labels_km+1==k).sum()})' for k in range(1, K_FINAL+1)], fontweight='bold')
#     for i in range(K_FINAL):
#         for j in range(len(cols_ord)):
#             val = tabla_pct.values[i, j]
#             ax.text(j, i, f'{val:.0f}%', ha='center', va='center', fontweight='bold', color='white' if val > 50 else 'black')
#     plt.savefig(out('COMPARACION_cluster_vs_patron_tucker_heatmap.png'), dpi=200, bbox_inches='tight')
#     plt.close()

# # ============================================================
# # 15. EXPORTAR Y EVALUAR (LÓGICA CON K_FINAL ACTUALIZADO)
# # ============================================================
# print("\nExportando resultados y evaluando estabilidad...")
# df_comp_cluster['silhouette_kmeans'] = silhouette_samples(X_casos, labels_km)
# df_comp_cluster.to_csv(out('CLUSTERING_COMPOSICION_por_caso.csv'), index=False, sep=';', encoding='utf-8-sig', decimal=',')

# aris_boot=[]
# for _ in range(100):
#     idx=resample(range(len(X_casos)))
#     km_b=KMeans(n_clusters=K_FINAL, n_init=10)
#     lbl_b=km_b.fit_predict(X_casos[idx])
#     aris_boot.append(adjusted_rand_score(labels_km[idx],lbl_b))
# print(f"  Estabilidad Bootstrap (k={K_FINAL}): ARI={np.mean(aris_boot):.3f} ± {np.std(aris_boot):.3f}")

# print("\n✓ PROCESO COMPLETADO EXITOSAMENTE.")

# # py -3.12 c:/Users/carme/OneDrive/Escritorio/M.UCM/TFM/1700MICRAS/CENTROIDE/FEATURE_ENGINEERING/clustering_features_eda2.py

"""

Clustering de CASOS basado en la COMPOSICION de patrones de tiles.

PIPELINE:
  1. Carga df_all_completo.csv
  2. Clusteriza los tiles individualmente por features biológicas (K_TILES clusters)
  3. Calcula composición porcentual de cada caso
  4. Clusteriza los casos: K-Means, Jerárquico, GMM, Spectral, HDBSCAN
  5. Genera figuras de caracterización y comparación de algoritmos
  6. Compara con Tucker (ARI)
  7. Genera mosaicos de crops por cluster


"""

DF_ALL_CSV    = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\1700MICRAS\CENTROIDE\FEATURE_ENGINEERING\new_eda2\df_all_completo.csv"
TUCKER_CSV    = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\1700MICRAS\CENTROIDE\FEATURE_ENGINEERING\TUCKER\CLUSTERING\CLUSTERING_resultados_por_caso.csv"
OUTPUT_FOLDER = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\1700MICRAS\CENTROIDE\FEATURE_ENGINEERING\CLUSTERING_EDA2"


K_TILES          = 5    # tipos de tile
K_RANGE_CASOS    = range(2, 9)
K_FORZADO_CASOS  = 4   
ZSCORE_OUTLIER   = 3.5

CZI_FOLDER    = r"\\imgserver\IMAGES\CONFOCAL\IA\crodriguezj\images"
PIXEL_SIZE_UM = 0.1723
ZOOM_CZI      = 0.15
CROP_SIZE_PX  = (220, 220)
MAX_CROPS_TOTAL = 50
MAX_CROPS_CASO  = 3

CLUSTER_COLORS = ['#E74C3C','#3498DB','#2ECC71','#F39C12','#9B59B6','#1ABC9C','#E67E22']
TILE_COLORS    = ['#E74C3C','#3498DB','#2ECC71','#F39C12','#9B59B6']

col_x1 = 'Object Info (tile) - Envelope left'
col_y1 = 'Object Info (tile) - Envelope top'
col_x2 = 'Object Info (tile) - Envelope right'
col_y2 = 'Object Info (tile) - Envelope bottom'


# IMPORTS

import os, warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA as skPCA
from sklearn.metrics import (silhouette_score, silhouette_samples,
                              davies_bouldin_score, calinski_harabasz_score,
                              adjusted_rand_score)
from sklearn.utils import resample
from scipy.cluster.hierarchy import dendrogram, linkage

try:
    import hdbscan
    HDBSCAN_OK = True
except ImportError:
    HDBSCAN_OK = False
    print("[AVISO] hdbscan no disponible. Instalar: pip install hdbscan --break-system-packages")

warnings.filterwarnings('ignore')
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
def out(f): return os.path.join(OUTPUT_FOLDER, f)

#  1. CARGAR DATOS

print("Cargando df_all_completo...")
df = pd.read_csv(DF_ALL_CSV)
print(f"  Shape: {df.shape}")
print(f"  Casos: {df['Case'].nunique()}")
print(f"  Tiles: {len(df):,}")

N_CASOS = df['Case'].nunique()

#  2. FEATURES BIOLOGICAS PARA CLUSTERING DE TILES

print("\nSeleccionando features biologicas...")

excluir = {col_x1, col_y1, col_x2, col_y2, 'Case', '_b1', '_b2',
           '_pat_dom', '_pat_intens', '_pat_color_g'}

cols_bio_ok = [c for c in df.columns
               if df[c].dtype in [float,'float64']
               and c not in excluir
               and not c.startswith('PC')
               and not c.startswith('_')
               and any(x in c for x in ['(Tumor)','(Muscle)','(No tumor/No muscle)','Interface'])
               and df[c].isna().mean() < 0.3
               and df[c].std() > 1e-6]

print(f"  Features válidas: {len(cols_bio_ok)}")

X_tiles_raw = df[cols_bio_ok].copy()
for c in cols_bio_ok:
    X_tiles_raw[c].fillna(X_tiles_raw[c].median(), inplace=True)

scaler_tiles = StandardScaler()
X_tiles      = scaler_tiles.fit_transform(X_tiles_raw.values)
print(f"  Matriz de tiles: {X_tiles.shape}")

#3. CLUSTERING DE TILES (PASO 1)

print(f"\n{'='*55}")
print(f"PASO 1: CLUSTERING DE TILES (k={K_TILES})")
print(f"{'='*55}")

km_tiles     = KMeans(n_clusters=K_TILES, random_state=42, n_init=30, max_iter=500)
labels_tiles = km_tiles.fit_predict(X_tiles)
df['cluster_tile'] = labels_tiles + 1

sil_tiles = silhouette_score(X_tiles, labels_tiles, sample_size=2000)
print(f"  Silhouette tiles: {sil_tiles:.3f}")
print(f"  Distribución: {dict(pd.Series(labels_tiles+1).value_counts().sort_index())}")

# Perfil biológico
df_tile_profile = pd.DataFrame(X_tiles, columns=cols_bio_ok)
df_tile_profile['cluster_tile'] = labels_tiles + 1
medias_tiles = df_tile_profile.groupby('cluster_tile')[cols_bio_ok].mean()
z_tiles      = medias_tiles.apply(lambda x: (x-x.mean())/(x.std()+1e-12), axis=0)

print("\n  Top features por cluster de tile:")
for ct in range(1, K_TILES+1):
    top5 = z_tiles.loc[ct].abs().nlargest(5).index.tolist()
    vals = [(f.split('(')[0].strip()[:18], f"{z_tiles.loc[ct,f]:.2f}") for f in top5]
    print(f"    Tile C{ct}: " + " | ".join([f"{n}={v}" for n,v in vals]))

# Figura perfil tiles
varianza_tiles = z_tiles.var(axis=0).sort_values(ascending=False)
top30_tiles    = varianza_tiles.head(30).index.tolist()

def nombre_corto(f):
    for rep,sust in [(' (No tumor/No muscle)','\n(Str)'),(' (Muscle)','\n(Mus)'),
                     (' (Tumor)','\n(Tum)'),('Interface Length ','Interf.'),
                     ('Intensity','Int.'),('Entropy 32bins ','Entr.'),('Area_div_','A/')]:
        f = f.replace(rep, sust)
    return f[:25]

fig, axes = plt.subplots(1, K_TILES,
                          figsize=(4*K_TILES, max(10, len(top30_tiles)*0.35)),
                          sharey=True, facecolor='white')
plt.subplots_adjust(wspace=0.02)
if K_TILES == 1: axes = [axes]
pos_y = np.arange(len(top30_tiles))
for ct_idx, ax in enumerate(axes):
    ct = ct_idx+1; col = TILE_COLORS[ct_idx]
    vals = z_tiles.loc[ct, top30_tiles].values
    ax.barh(pos_y, vals,
            color=['#C0392B' if v>0 else '#2980B9' for v in vals],
            edgecolor='none', height=0.7, alpha=0.88)
    ax.axvline(0, color='black', lw=1); ax.set_xlim(-3, 3)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.set_title(f'Tile C{ct}\n({(labels_tiles+1==ct).sum():,} tiles)',
                 fontsize=11, fontweight='bold', color=col)
    ax.set_xlabel('Z-score', fontsize=8)
    if ct == 1:
        ax.set_yticks(pos_y)
        ax.set_yticklabels([nombre_corto(f) for f in top30_tiles], fontsize=7)
    else:
        ax.tick_params(left=False)
    for spine in ax.spines.values():
        spine.set_edgecolor(col); spine.set_linewidth(2)
plt.suptitle(f'Perfil Biológico de Clusters de Tiles (k={K_TILES})',
             fontsize=13, fontweight='bold', y=0.98)
plt.savefig(out('00_perfil_clusters_tiles.png'), dpi=180, bbox_inches='tight', facecolor='white')
plt.close()


#4. COMPOSICION PORCENTUAL POR CASO (PASO 2)

print(f"\n{'='*55}")
print(f"PASO 2: COMPOSICION DE TILES POR CASO")


df_comp = (df.groupby('Case')['cluster_tile']
             .value_counts(normalize=True)
             .unstack(fill_value=0))
for ct in range(1, K_TILES+1):
    if ct not in df_comp.columns: df_comp[ct] = 0.0
df_comp = df_comp[[ct for ct in range(1, K_TILES+1)]].copy()
df_comp.columns = [f'pct_TileC{ct}' for ct in range(1, K_TILES+1)]
df_comp = df_comp.reset_index()
pct_cols = [f'pct_TileC{ct}' for ct in range(1, K_TILES+1)]

print(f"  Casos: {len(df_comp)}")
print(df_comp[pct_cols].describe().round(3).to_string())

df_comp.to_csv(out('COMPOSICION_tiles_por_caso.csv'),
               index=False, sep=';', encoding='utf-8-sig', decimal=',')


# Heatmap composición
fig, ax = plt.subplots(figsize=(N_CASOS*0.3+2, 5), facecolor='white')
data_hm  = df_comp.set_index('Case')[pct_cols].T
caso_dom = df_comp.set_index('Case')[pct_cols].idxmax(axis=1)
orden_c  = df_comp.set_index('Case').assign(_d=caso_dom).sort_values('_d').index
sns.heatmap(data_hm[orden_c], ax=ax, cmap='YlOrRd', vmin=0, vmax=1,
            xticklabels=True,
            yticklabels=[f'Tile C{ct}' for ct in range(1, K_TILES+1)],
            cbar_kws={'label':'% tiles','shrink':0.6},
            linewidths=0.2, linecolor='#DDDDDD')
ax.set_xticklabels(ax.get_xticklabels(), rotation=75, ha='right', fontsize=5)
for label, col in zip(ax.get_yticklabels(), TILE_COLORS): label.set_color(col)
ax.set_title(f'Composición de Tipos de Tile por Caso (n={N_CASOS})',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(out('01_composicion_heatmap.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()


# 5. DETECCION DE OUTLIERS

print(f"\n{'='*55}")
print(f"DETECCION DE OUTLIERS (z-score > {ZSCORE_OUTLIER})")


z_matrix  = np.abs(stats.zscore(df_comp[pct_cols].values))
mask_ok   = (z_matrix < ZSCORE_OUTLIER).all(axis=1)
n_outliers = (~mask_ok).sum()

if n_outliers > 0:
    outlier_casos = df_comp.loc[~mask_ok, 'Case'].tolist()
    for caso in outlier_casos:
        idx = df_comp.index[df_comp['Case']==caso][0]
        zs  = z_matrix[idx]
        top = sorted(zip(pct_cols, zs), key=lambda x: -x[1])[:3]
        top_str = ', '.join([f"{c}(z={z:.1f})" for c,z in top if z>ZSCORE_OUTLIER])
        print(f"  ⚠ OUTLIER: {caso} → {top_str}")
    print(f"\n  Total outliers: {n_outliers}")
    df_comp_cluster = df_comp[mask_ok].copy().reset_index(drop=True)
else:
    print(f"  NO outliers")
    outlier_casos   = []
    df_comp_cluster = df_comp.copy()

casos_cluster = df_comp_cluster['Case'].tolist()
print(f"  Casos para clustering: {len(casos_cluster)}")

scaler_casos = StandardScaler()
X_casos      = scaler_casos.fit_transform(df_comp_cluster[pct_cols].values)

#  6. METRICAS PARA k OPTIMO

print(f"\n{'='*55}")
print(f"PASO 3: CLUSTERING DE CASOS")


metricas = {'k':[],'inercia':[],'silhouette':[],'davies_bouldin':[],'calinski':[]}
for k in K_RANGE_CASOS:
    if k >= len(casos_cluster): continue
    km     = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_casos)
    metricas['k'].append(k)
    metricas['inercia'].append(km.inertia_)
    metricas['silhouette'].append(silhouette_score(X_casos, labels))
    metricas['davies_bouldin'].append(davies_bouldin_score(X_casos, labels))
    metricas['calinski'].append(calinski_harabasz_score(X_casos, labels))

df_met  = pd.DataFrame(metricas)
best_s  = int(df_met.loc[df_met['silhouette'].idxmax(), 'k'])
best_db = int(df_met.loc[df_met['davies_bouldin'].idxmin(), 'k'])
best_ch = int(df_met.loc[df_met['calinski'].idxmax(), 'k'])

print(df_met.to_string(index=False))
print(f"\n  Silhouette→k={best_s} | Davies-Bouldin→k={best_db} | Calinski→k={best_ch}")

df_met.to_csv(out('METRICAS_k_optimo_casos.csv'),
              index=False, sep=';', encoding='utf-8-sig', decimal=',')

fig, axes = plt.subplots(2, 2, figsize=(13,9), facecolor='white')
fig.suptitle('Métricas para k óptimo — Clustering por Composición de Tiles',
             fontsize=13, fontweight='bold')
pares = [(axes[0,0],'inercia','#E74C3C','Inercia → buscar codo',None),
         (axes[0,1],'silhouette','#3498DB','Silhouette → mayor = mejor',best_s),
         (axes[1,0],'davies_bouldin','#2ECC71','Davies-Bouldin → menor mejor',best_db),
         (axes[1,1],'calinski','#9B59B6','Calinski → mayor = mejor',best_ch)]
for ax,met,col,tit,bk in pares:
    ax.plot(df_met['k'], df_met[met], 'o-', color=col, lw=2, ms=8)
    ax.set_title(tit, fontsize=10, fontweight='bold')
    ax.set_xlabel('k'); ax.grid(alpha=0.3); ax.set_xticks(list(df_met['k']))
    if bk:
        ax.axvline(bk, color=col, linestyle='--', alpha=0.7, label=f'mejor k={bk}')
        ax.legend(fontsize=8)
    if K_FORZADO_CASOS:
        ax.axvline(K_FORZADO_CASOS, color='black', linestyle=':', alpha=0.6)
        ax.legend(fontsize=8)
fig.text(0.5, 0.01,
         f'Sil→k={best_s} | DB→k={best_db} | CH→k={best_ch} | K_FORZADO={K_FORZADO_CASOS}',
         ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#ECF0F1', alpha=0.8))
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(out('02_metricas_k_optimo.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()


# Seleccionar K_FINAL
if K_FORZADO_CASOS is not None:
    K_FINAL = K_FORZADO_CASOS
    print(f"\nUsando K_FINAL = {K_FINAL}  (forzado)")
else:
    K_FINAL = Counter([best_s, best_db, best_ch]).most_common(1)[0][0]
    print(f"\nUsando K_FINAL = {K_FINAL}  (votación: Sil={best_s}, DB={best_db}, CH={best_ch})")

#  7. APLICAR TODOS LOS ALGORITMOS

# K-Means
km_final  = KMeans(n_clusters=K_FINAL, random_state=42, n_init=50)
labels_km = km_final.fit_predict(X_casos)

# Jerárquico Ward
hc_final  = AgglomerativeClustering(n_clusters=K_FINAL, linkage='ward')
labels_hc = hc_final.fit_predict(X_casos)

# GMM
gmm_final = GaussianMixture(n_components=K_FINAL, covariance_type='full',
                             random_state=42, n_init=20)
gmm_final.fit(X_casos)
labels_gm = gmm_final.predict(X_casos)
proba_gm  = gmm_final.predict_proba(X_casos)

# Spectral Clustering
print("  Aplicando Spectral Clustering...")
spec        = SpectralClustering(n_clusters=K_FINAL, random_state=42,
                                  affinity='nearest_neighbors', n_neighbors=10)
labels_spec = spec.fit_predict(X_casos)

# HDBSCAN (sin k fijo)
if HDBSCAN_OK:
    
    clusterer   = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3, metric='euclidean')
    labels_hdb  = clusterer.fit_predict(X_casos)
    n_cl_hdb    = len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0)
    n_noise_hdb = (labels_hdb == -1).sum()
    print(f"  HDBSCAN encontró: {n_cl_hdb} clusters, {n_noise_hdb} puntos de ruido")
    if n_cl_hdb > 1:
        mask_v  = labels_hdb != -1
        sil_hdb = silhouette_score(X_casos[mask_v], labels_hdb[mask_v])
        print(f"  Silhouette HDBSCAN: {sil_hdb:.3f}")
        print(f"  Distribución: {dict(pd.Series(labels_hdb[mask_v]).value_counts().sort_index())}")
else:
    labels_hdb  = np.full(len(X_casos), -1)
    n_cl_hdb    = 0
    n_noise_hdb = len(X_casos)
    sil_hdb     = 0.0

# Guardar en df
df_comp_cluster['cluster_kmeans']     = labels_km + 1
df_comp_cluster['cluster_jerarquico'] = labels_hc + 1
df_comp_cluster['cluster_gmm']        = labels_gm + 1
df_comp_cluster['cluster_spectral']   = labels_spec + 1
df_comp_cluster['cluster_hdbscan']    = labels_hdb   # -1 = ruido

# Resumen de cada método
print("\n  Resumen por método:")
for metodo, lbl in [('K-Means',    labels_km+1),
                    ('Jerárquico', labels_hc+1),
                    ('GMM',        labels_gm+1),
                    ('Spectral',   labels_spec+1)]:
    dist = dict(pd.Series(lbl).value_counts().sort_index())
    sil  = silhouette_score(X_casos, lbl)
    print(f"  {metodo:12s}: {dist}  Sil={sil:.3f}")

# 8. DENDROGRAMA

Z_link = linkage(X_casos, method='ward')
fig, ax = plt.subplots(figsize=(max(18, len(casos_cluster)*0.28), 8), facecolor='white')
dendrogram(Z_link, labels=list(casos_cluster), ax=ax,
           color_threshold=0.7*max(Z_link[:,2]),
           leaf_rotation=75, leaf_font_size=6, above_threshold_color='#888888')
ax.set_title('Dendrograma Jerárquico (Ward) — Composición de Tipos de Tile por Caso'
             + (f'\nOutliers excluidos: {outlier_casos}' if n_outliers>0 else ''),
             fontsize=12, fontweight='bold')
ax.set_ylabel('Distancia (Ward)'); ax.set_xlabel('Casos')
ax.grid(axis='y', alpha=0.3)
for ki, k in enumerate([2,3,4,5]):
    if k < len(casos_cluster):
        h = Z_link[-k+1, 2]
        ax.axhline(h, color=CLUSTER_COLORS[ki], lw=1.5, linestyle='--',
                   alpha=0.8, label=f'k={k} (h={h:.3f})')
ax.legend(fontsize=9, loc='upper right')
plt.tight_layout()
plt.savefig(out('03_dendrograma.png'), dpi=180, bbox_inches='tight', facecolor='white')
plt.close()

# 9. PROYECCION 2D — K-Means + Jerárquico + GMM

pca2   = skPCA(n_components=2)
X_2d   = pca2.fit_transform(X_casos)
var_2d = pca2.explained_variance_ratio_

fig, axes = plt.subplots(1, 3, figsize=(18,6), facecolor='white')
fig.suptitle(f'Proyección 2D — Clustering por Composición de Tiles (k={K_FINAL})\n'
             f'Dim1={var_2d[0]*100:.1f}% | Dim2={var_2d[1]*100:.1f}% varianza',
             fontsize=12, fontweight='bold')
for ax, titulo, etq in zip(axes,
        ['K-Means','Jerárquico (Ward)','GMM'],
        [labels_km+1, labels_hc+1, labels_gm+1]):
    for k in range(1, K_FINAL+1):
        mask = etq == k
        ax.scatter(X_2d[mask,0], X_2d[mask,1], c=CLUSTER_COLORS[k-1], s=90,
                   alpha=0.85, edgecolors='white', lw=0.8, zorder=3,
                   label=f'C{k} (n={mask.sum()})')
        if mask.sum() > 0:
            ax.scatter(X_2d[mask,0].mean(), X_2d[mask,1].mean(),
                       c=CLUSTER_COLORS[k-1], s=220, marker='*',
                       edgecolors='black', lw=1.5, zorder=5)
        for idx in np.where(mask)[0]:
            ax.annotate(casos_cluster[idx], (X_2d[idx,0], X_2d[idx,1]),
                        fontsize=3.5, alpha=0.6, ha='center', va='bottom')
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    ax.set_xlabel(f'Dim1 ({var_2d[0]*100:.1f}%)')
    ax.set_ylabel(f'Dim2 ({var_2d[1]*100:.1f}%)')
    ax.legend(fontsize=9, framealpha=0.9); ax.grid(alpha=0.25); ax.set_facecolor('#FAFAFA')
plt.tight_layout()
plt.savefig(out('04_proyeccion_2d.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()


# 10. FIGURA COMPARATIVA 4 ALGORITMOS (NUEVA)


fig, axes = plt.subplots(1, 4, figsize=(24,6), facecolor='white')
fig.suptitle('Comparación de Algoritmos de Clustering — Composición de Tiles\n'
             f'Dim1={var_2d[0]*100:.1f}% | Dim2={var_2d[1]*100:.1f}% varianza explicada',
             fontsize=13, fontweight='bold')

algoritmos = [
    ('K-Means',    labels_km+1),
    ('Spectral',   labels_spec+1),
    ('Jerárquico', labels_hc+1),
    ('HDBSCAN',    labels_hdb),
]

for ax, (titulo, etq) in zip(axes, algoritmos):
    vals_unicos = sorted(set(etq))
    n_cl = len(vals_unicos) - (1 if -1 in vals_unicos else 0)

    for k in vals_unicos:
        mask  = etq == k
        color = '#AAAAAA' if k == -1 else CLUSTER_COLORS[(k-1) % len(CLUSTER_COLORS)]
        label = 'Ruido' if k == -1 else f'C{k} (n={mask.sum()})'
        ax.scatter(X_2d[mask,0], X_2d[mask,1], c=color, s=80,
                   alpha=0.85, edgecolors='white', lw=0.8, zorder=3, label=label)
        if k != -1 and mask.sum() > 0:
            ax.scatter(X_2d[mask,0].mean(), X_2d[mask,1].mean(),
                       c=color, s=200, marker='*', edgecolors='black', lw=1.5, zorder=5)

    mask_v  = etq != -1
    sil_val = silhouette_score(X_casos[mask_v], etq[mask_v]) if n_cl > 1 else 0
    ax.set_title(f'{titulo}\n({n_cl} clusters | Sil={sil_val:.3f})',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel(f'Dim1'); ax.set_ylabel(f'Dim2')
    ax.legend(fontsize=7, framealpha=0.9, loc='best')
    ax.grid(alpha=0.25); ax.set_facecolor('#FAFAFA')

plt.tight_layout()
plt.savefig(out('04b_comparacion_algoritmos.png'),
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()


#  11. PERFIL DE COMPOSICION POR CLUSTER



medias_comp = df_comp_cluster.groupby('cluster_kmeans')[pct_cols].mean()

fig, ax = plt.subplots(figsize=(K_FINAL*2+3, 6), facecolor='white')
x_pos = np.arange(K_TILES); bar_w = 0.8/K_FINAL
for cl_idx in range(K_FINAL):
    cl = cl_idx+1
    if cl not in medias_comp.index: continue
    vals = medias_comp.loc[cl, pct_cols].values * 100
    ax.bar(x_pos+cl_idx*bar_w-(K_FINAL-1)*bar_w/2, vals, bar_w*0.9,
           color=CLUSTER_COLORS[cl_idx], alpha=0.85,
           label=f'Cluster {cl} (n={(labels_km+1==cl).sum()})',
           edgecolor='white', lw=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'Tile C{ct}' for ct in range(1,K_TILES+1)], fontsize=11, fontweight='bold')
for tick,col in zip(ax.get_xticklabels(), TILE_COLORS): tick.set_color(col)
ax.set_ylabel('% medio de tiles (%)'); ax.set_ylim(0,100)
ax.set_title('Composición de Tipos de Tile por Cluster de Caso (K-Means)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.9); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(out('05_composicion_por_cluster_caso.png'),
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()


# Heatmap composición 
fig, ax = plt.subplots(figsize=(K_FINAL*1.8+2, K_TILES*0.8+2), facecolor='white')
im = ax.imshow(medias_comp[pct_cols].values.T*100, cmap='YlOrRd',
               vmin=0, vmax=100, aspect='auto')
ax.set_xticks(range(K_FINAL))
ax.set_xticklabels([f'Cluster {cl}\n(n={(labels_km+1==cl).sum()})'
                    for cl in range(1,K_FINAL+1)], fontsize=10, fontweight='bold')
for tick,col in zip(ax.get_xticklabels(), CLUSTER_COLORS): tick.set_color(col)
ax.set_yticks(range(K_TILES))
ax.set_yticklabels([f'Tile C{ct}' for ct in range(1,K_TILES+1)],
                   fontsize=10, fontweight='bold')
for tick,col in zip(ax.get_yticklabels(), TILE_COLORS): tick.set_color(col)
for i in range(K_TILES):
    for j in range(K_FINAL):
        val = medias_comp[pct_cols].values.T[i,j]*100
        ax.text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=11, fontweight='bold',
                color='white' if val>50 else 'black')
plt.colorbar(im, ax=ax, label='% medio de tiles', shrink=0.6)
ax.set_title(' Composición media de tiles por cluster\n'
             'Cada celda = % medio de tiles de ese tipo en los casos del cluster',
             fontsize=12, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(out('05b_composicion_heatmap.png'),
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()


# 12. CONCORDANCIA ENTRE METODOS

ari_km_hc   = adjusted_rand_score(labels_km, labels_hc)
ari_km_gm   = adjusted_rand_score(labels_km, labels_gm)
ari_hc_gm   = adjusted_rand_score(labels_hc, labels_gm)
ari_km_spec = adjusted_rand_score(labels_km, labels_spec)

fig, axes = plt.subplots(1, 4, figsize=(22,5), facecolor='white')
fig.suptitle(f'Concordancia entre Métodos (k={K_FINAL}) — '
             f'KM-HC={ari_km_hc:.3f} | KM-GMM={ari_km_gm:.3f} | '
             f'HC-GMM={ari_hc_gm:.3f} | KM-Spec={ari_km_spec:.3f}',
             fontsize=11, fontweight='bold')
for ax,(n1,l1,n2,l2,ari) in zip(axes,[
        ('K-Means',    labels_km+1,   'Jerarquico', labels_hc+1,   ari_km_hc),
        ('K-Means',    labels_km+1,   'GMM',        labels_gm+1,   ari_km_gm),
        ('Jerarquico', labels_hc+1,   'GMM',        labels_gm+1,   ari_hc_gm),
        ('K-Means',    labels_km+1,   'Spectral',   labels_spec+1, ari_km_spec)]):
    mat = np.zeros((K_FINAL, K_FINAL), dtype=int)
    for a,b in zip(l1,l2): mat[a-1,b-1] += 1
    sns.heatmap(mat, ax=ax, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'{n2} C{k}' for k in range(1,K_FINAL+1)],
                yticklabels=[f'{n1} C{k}' for k in range(1,K_FINAL+1)])
    ax.set_title(f'{n1} vs {n2}\nARI={ari:.3f}', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(out('06_concordancia_metodos.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

#  13. PERFIL FEATURES BIOLOGICAS POR CLUSTER



df_feat = df[df['Case'].isin(casos_cluster)][['Case']+cols_bio_ok].merge(
    df_comp_cluster[['Case','cluster_kmeans']], on='Case', how='left')
df_fc   = df_feat.groupby('cluster_kmeans')[cols_bio_ok].mean().T
df_fc.columns = [f'C{k}' for k in df_fc.columns]
df_fn   = df_fc.copy()
for feat in df_fn.index:
    row = df_fn.loc[feat]; rng = row.max()-row.min()
    df_fn.loc[feat] = (row-row.min())/rng if rng>1e-10 else 0.5

grupos = {
    'MUSCULO':  [c for c in cols_bio_ok if '(Muscle)' in c],
    'TUMOR':    [c for c in cols_bio_ok if '(Tumor)' in c],
    'ESTROMA':  [c for c in cols_bio_ok if '(No tumor/No muscle)' in c],
    'INTERFAZ': [c for c in cols_bio_ok if 'Interface' in c],
}
def nc(f):
    for r,s in [(' (No tumor/No muscle)',''),(' (Muscle)',''),(' (Tumor)',''),
                ('Interface Length ','Interf. '),('Area_div_','A/'),
                ('Entropy 32bins ','Entr. '),('Intensity','Int.')]:
        f = f.replace(r,s)
    return f

orden=[]; etq=[]; sep={}; pos=0
for gnom,feats in grupos.items():
    if not feats: continue
    fs=sorted(feats); sep[gnom]=(pos,pos+len(fs))
    for f in fs: orden.append(f); etq.append(nc(f))
    pos+=len(fs)

pos_y=np.arange(len(orden)); BG=['#F0F0F0','#E8E8E8','#F0F0F0','#E8E8E8']
fig,axes=plt.subplots(1,K_FINAL,figsize=(5*K_FINAL,max(16,len(orden)*0.22+2)),
                      sharey=True,facecolor='white')
plt.subplots_adjust(wspace=0.03,left=0.22,right=0.97,top=0.93,bottom=0.03)
if K_FINAL==1: axes=[axes]
for k_idx,ax in enumerate(axes):
    k=k_idx+1; col=CLUSTER_COLORS[k_idx]; cn=f'C{k}'
    if cn not in df_fn.columns: ax.set_visible(False); continue
    vals=df_fn.reindex(orden)[cn].values
    for gi,(gnom,(g0,g1)) in enumerate(sep.items()):
        ax.axhspan(g0-0.5,g1-0.5,facecolor=BG[gi%4],alpha=0.5,zorder=0)
    ax.barh(pos_y,vals,color=col,edgecolor='none',height=0.75,alpha=0.88,zorder=2)
    ax.axvline(0.5,color='gray',lw=0.8,alpha=0.6,linestyle='--',zorder=3)
    for gnom,(g0,g1) in sep.items():
        ax.axhline(g0-0.5,color='black',lw=1.5,alpha=0.7,zorder=4)
        if k==1:
            ax.text(-0.18,(g0+g1)/2-0.5,gnom,ha='right',va='center',
                    fontsize=8,fontweight='bold',transform=ax.get_yaxis_transform(),
                    rotation=90)
    nk=(df_comp_cluster['cluster_kmeans']==k).sum()
    ax.set_xlim(0,1.05); ax.set_xticks([0,0.5,1.0])
    ax.set_xticklabels(['0','0.5','1'],fontsize=7)
    ax.grid(axis='x',linestyle='--',alpha=0.25,zorder=1)
    ax.set_title(f'Cluster {k}\n(n={nk} casos)',fontsize=12,fontweight='bold',color=col,pad=6)
    ax.set_xlabel('[0=min, 1=max]',fontsize=7)
    for spine in ax.spines.values(): spine.set_edgecolor(col); spine.set_linewidth(2.5)
    if k==1: ax.set_yticks(pos_y); ax.set_yticklabels(etq,fontsize=6.5)
    else: ax.tick_params(left=False)
plt.suptitle(f'PERFIL MORFOLÓGICO POR CLUSTER — Composición de Tiles k={K_FINAL}',
             fontsize=13,fontweight='bold',y=0.98)
plt.savefig(out('07_perfil_features_por_cluster.png'),
            dpi=180,bbox_inches='tight',facecolor='white')
plt.close()
print("  ✓ 07_perfil_features_por_cluster.png")

# # ============================================================
# # 14. COMPARACION CON TUCKER
# # ============================================================

# tucker_csv_path = Path(TUCKER_CSV)
# aris = {}
# if tucker_csv_path.exists():
#     print("\nComparando con Tucker...")
#     df_tucker = pd.read_csv(TUCKER_CSV, sep=';')
#     df_comp2  = df_comp_cluster[['Case','cluster_kmeans']].merge(
#         df_tucker[['Case']+[c for c in df_tucker.columns if c.startswith('km_')]],
#         on='Case', how='inner')
#     for col in [c for c in df_comp2.columns if c not in ('Case','cluster_kmeans')]:
#         try: aris[col] = adjusted_rand_score(df_comp2['cluster_kmeans'].values, df_comp2[col].values)
#         except: pass
#     for col,ari in aris.items():
#         nivel = 'ALTO' if ari>0.6 else ('MEDIO' if ari>0.3 else 'BAJO')
#         print(f"    {col}: ARI={ari:.3f} ({nivel})")

#     fig,axes=plt.subplots(1,len(aris)+1,figsize=(6*(len(aris)+1),6),facecolor='white')
#     fig.suptitle(f'Composición Tiles vs Tucker (k={K_FINAL})',fontsize=13,fontweight='bold')
#     ax=axes[0]
#     for k in range(1,K_FINAL+1):
#         mask=labels_km+1==k
#         ax.scatter(X_2d[mask,0],X_2d[mask,1],c=CLUSTER_COLORS[k-1],s=90,
#                    alpha=0.85,edgecolors='white',lw=0.8,label=f'C{k}',zorder=3)
#     ax.set_title(f'Clusters CompTile (k={K_FINAL})'); ax.legend(fontsize=9)
#     ax.grid(alpha=0.25); ax.set_facecolor('#FAFAFA')
#     for ax_i,(col,ari) in enumerate(aris.items()):
#         ax=axes[ax_i+1]; n_tck=df_comp2[col].nunique()
#         mat=np.zeros((K_FINAL,n_tck),dtype=int)
#         for fd,tk in zip(df_comp2['cluster_kmeans'],df_comp2[col]):
#             mat[int(fd)-1,int(tk)-1]+=1
#         sns.heatmap(mat,ax=ax,annot=True,fmt='d',cmap='YlOrRd',
#                     xticklabels=[f'Tucker C{k}' for k in range(1,n_tck+1)],
#                     yticklabels=[f'CT C{k}' for k in range(1,K_FINAL+1)])
#         nivel='ALTO' if ari>0.6 else ('MEDIO' if ari>0.3 else 'BAJO')
#         ax.set_title(f'CT(k={K_FINAL}) vs {col}\nARI={ari:.3f} ({nivel})',fontsize=10,fontweight='bold')
#     plt.tight_layout()
#     plt.savefig(out('08_comparacion_tucker.png'),dpi=200,bbox_inches='tight',facecolor='white')
#     plt.close()
#     print("  ✓ 08_comparacion_tucker.png")

#     df_cruce=df_comp_cluster[['Case','cluster_kmeans']].merge(
#         df_tucker[['Case','km_k2','km_k5']],on='Case',how='inner')
#     df_cruce.columns=['Case','cluster_composicion','tucker_k2','tucker_k5']
#     df_cruce.to_csv(out('CORRESPONDENCIA_composicion_tucker.csv'),
#                     index=False,sep=';',encoding='utf-8-sig',decimal=',')
#     print("  ✓ CORRESPONDENCIA_composicion_tucker.csv")

# COMPARACION CON PATRONES TUCKER (no con clusters Tucker)

TUCKER_TILES_CSV   = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\1700MICRAS\CENTROIDE\FEATURE_ENGINEERING\TUCKER\new_eda2\TUCKER_tiles_patron_dom.csv"
TUCKER_SCORES_CSV  = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\1700MICRAS\CENTROIDE\FEATURE_ENGINEERING\TUCKER\new_eda2\TUCKER_contribuciones_por_caso.csv"

tucker_tiles_path  = Path(TUCKER_TILES_CSV)
tucker_scores_path = Path(TUCKER_SCORES_CSV)

if tucker_tiles_path.exists() and tucker_scores_path.exists():



    df_tucker_scores = pd.read_csv(TUCKER_SCORES_CSV, sep=';')
    pat_tucker_cols  = [c for c in df_tucker_scores.columns if c.startswith('Patron_')]
    N_PAT_T = len(pat_tucker_cols)
    COLORS_PAT_T = ['#2ECC71','#F39C12','#E74C3C','#9B59B6','#3498DB']

    print(f"  Patrones Tucker disponibles: {pat_tucker_cols}")

    # Merge: cluster de composición + scores Tucker por caso
    df_merge = df_comp_cluster[['Case','cluster_kmeans']].merge(
        df_tucker_scores[['Case','patron_dominante'] + pat_tucker_cols],
        on='Case', how='inner'
    )
    print(f"  Casos en común: {len(df_merge)}")

    # ── 1. Tabla: distribución del patrón dominante Tucker por cluster ──
    tabla_dom = pd.crosstab(
        df_merge['cluster_kmeans'],
        df_merge['patron_dominante'],
        margins=True
    )
    print("\n  Patrón Tucker dominante por cluster de composición:")
    print(tabla_dom.to_string())

    tabla_dom.to_csv(
        out('COMPARACION_cluster_vs_patron_tucker_dominante.csv'),
        sep=';', encoding='utf-8-sig'
    )

    # ── 2. Heatmap: % patrón Tucker dominante por cluster ──
    tabla_pct = pd.crosstab(
        df_merge['cluster_kmeans'],
        df_merge['patron_dominante'],
        normalize='index'
    ) * 100

    fig, ax = plt.subplots(
        figsize=(N_PAT_T*1.5+2, K_FINAL*0.9+2), facecolor='white'
    )
    # Reordenar columnas por número de patrón
    cols_ord = sorted(tabla_pct.columns,
                      key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
    tabla_pct = tabla_pct.reindex(columns=cols_ord, fill_value=0)

    im = ax.imshow(tabla_pct.values, cmap='YlOrRd', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(cols_ord)))
    ax.set_xticklabels(
        [f"P{c.split('_')[1]}" for c in cols_ord],
        fontsize=12, fontweight='bold'
    )
    for tick, col in zip(ax.get_xticklabels(), COLORS_PAT_T):
        tick.set_color(col)
    ax.set_yticks(range(K_FINAL))
    ax.set_yticklabels(
        [f'Cluster {k}\n(n={(labels_km+1==k).sum()})' for k in range(1, K_FINAL+1)],
        fontsize=10, fontweight='bold'
    )
    for tick, col in zip(ax.get_yticklabels(), CLUSTER_COLORS):
        tick.set_color(col)
    for i in range(K_FINAL):
        for j in range(len(cols_ord)):
            val = tabla_pct.values[i, j]
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='white' if val > 50 else 'black')
    plt.colorbar(im, ax=ax, label='% de casos', shrink=0.6)
    ax.set_title(
        '% de casos de cada cluster con cada Patrón Tucker dominante\n'
        'Filas = clusters de composición de tiles | '
        'Columnas = patrón Tucker dominante del caso',
        fontsize=12, fontweight='bold', pad=10
    )
    plt.tight_layout()
    plt.savefig(out('COMPARACION_cluster_vs_patron_tucker_heatmap.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ COMPARACION_cluster_vs_patron_tucker_heatmap.png")

    # ── 3. Score medio Tucker por cluster ──
    scores_por_cluster = df_merge.groupby('cluster_kmeans')[pat_tucker_cols].mean()
    print("\n  Score medio Tucker por cluster de composición:")
    print(scores_por_cluster.round(3).to_string())

    fig, ax = plt.subplots(
        figsize=(K_FINAL*2+3, 6), facecolor='white'
    )
    x_pos = np.arange(N_PAT_T)
    bar_w = 0.8 / K_FINAL
    for cl_idx in range(K_FINAL):
        cl = cl_idx + 1
        if cl not in scores_por_cluster.index: continue
        vals = scores_por_cluster.loc[cl, pat_tucker_cols].values
        ax.bar(
            x_pos + cl_idx*bar_w - (K_FINAL-1)*bar_w/2,
            vals, bar_w*0.9,
            color=CLUSTER_COLORS[cl_idx], alpha=0.85,
            label=f'Cluster {cl} (n={(labels_km+1==cl).sum()})',
            edgecolor='white', lw=0.5
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f'P{i+1}' for i in range(N_PAT_T)],
        fontsize=12, fontweight='bold'
    )
    for tick, col in zip(ax.get_xticklabels(), COLORS_PAT_T):
        tick.set_color(col)
    ax.set_ylabel('Score Tucker medio (normalizado)')
    ax.set_title(
        'Score medio de cada Patrón Tucker por Cluster de Composición\n'
        'Muestra a qué patrones Tucker se asocia cada cluster',
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out('COMPARACION_score_tucker_por_cluster.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ COMPARACION_score_tucker_por_cluster.png")

    # ── 4. Radar: perfil Tucker de cada cluster ──
    angles = [n/float(N_PAT_T)*2*np.pi for n in range(N_PAT_T)] + [0]
    fig, axes = plt.subplots(
        1, K_FINAL, figsize=(4.5*K_FINAL, 5),
        subplot_kw=dict(polar=True), facecolor='white'
    )
    fig.suptitle(
        'Perfil de Patrones Tucker por Cluster de Composición\n'
        'Cada eje = score medio del patrón Tucker en ese cluster',
        fontsize=12, fontweight='bold'
    )
    if K_FINAL == 1: axes = [axes]
    for k_idx, ax in enumerate(axes):
        k = k_idx + 1; col = CLUSTER_COLORS[k_idx]
        if k not in scores_por_cluster.index:
            ax.set_visible(False); continue
        vals = scores_por_cluster.loc[k, pat_tucker_cols].tolist()
        vals_cierre = vals + [vals[0]]
        ax.plot(angles, vals_cierre, color=col, lw=2.5)
        ax.fill(angles, vals_cierre, color=col, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            [f'P{i+1}' for i in range(N_PAT_T)],
            fontsize=10, fontweight='bold'
        )
        for tick, c in zip(ax.get_xticklabels(), COLORS_PAT_T):
            tick.set_color(c)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25','0.5','0.75','1.0'], fontsize=7, color='gray')
        ax.grid(color='gray', alpha=0.3)
        nk = (labels_km+1 == k).sum()
        ax.set_title(
            f'Cluster {k}\n(n={nk} casos)',
            fontsize=11, fontweight='bold', color=col, pad=15
        )
    plt.tight_layout()
    #plt.savefig(out('COMPARACION_radar_tucker_por_cluster.png'),
                #dpi=200, bbox_inches='tight', facecolor='white')
    #plt.close()
    #print("  ✓ COMPARACION_radar_tucker_por_cluster.png")

    # ── 5. Tabla exportable completa caso a caso ──
    df_export_comp = df_merge[
        ['Case', 'cluster_kmeans', 'patron_dominante'] + pat_tucker_cols
    ].copy()
    df_export_comp.columns = (
        ['Case', 'cluster_composicion', 'patron_tucker_dominante'] +
        [f'score_{c}' for c in pat_tucker_cols]
    )
    df_export_comp.to_csv(
        out('TABLA_cluster_vs_tucker_por_caso.csv'),
        index=False, sep=';', encoding='utf-8-sig', decimal=','
    )
    

    



#  15. EXPORTAR CSV


for ki in range(K_FINAL):
    df_comp_cluster[f'gmm_proba_C{ki+1}'] = proba_gm[:,ki]
df_comp_cluster['gmm_max_proba']     = proba_gm.max(axis=1)
df_comp_cluster['silhouette_kmeans'] = silhouette_samples(X_casos, labels_km)

cols_exp = (['Case','cluster_kmeans','cluster_jerarquico','cluster_gmm',
             'cluster_spectral','cluster_hdbscan',
             'gmm_max_proba','silhouette_kmeans'] + pct_cols +
            [f'gmm_proba_C{ki+1}' for ki in range(K_FINAL)])

if n_outliers > 0:
    df_out = df_comp[~mask_ok][['Case']+pct_cols].copy()
    for c in ['cluster_kmeans','cluster_jerarquico','cluster_gmm',
              'cluster_spectral','cluster_hdbscan']: df_out[c] = 0
    df_out['gmm_max_proba']=np.nan; df_out['silhouette_kmeans']=np.nan
    for ki in range(K_FINAL): df_out[f'gmm_proba_C{ki+1}']=np.nan
    for c in cols_exp:
        if c not in df_out.columns: df_out[c]=np.nan
    df_export=pd.concat([df_comp_cluster[cols_exp],df_out[cols_exp]],ignore_index=True)
    print(f"  ⚠ {n_outliers} outlier(s) con cluster=0: {outlier_casos}")
else:
    df_export=df_comp_cluster[cols_exp]

df_export.to_csv(out('CLUSTERING_COMPOSICION_por_caso.csv'),
                 index=False,sep=';',encoding='utf-8-sig',decimal=',')


#  16. METRICAS DE EVALUACION

print("\n"+"="*55+"\nMETRICAS DE EVALUACION\n"+"="*55)

sil_samples = silhouette_samples(X_casos, labels_km)
print(f"\nMETRICAS K-Means (k={K_FINAL}, {len(casos_cluster)} casos):")
print(f"  Silhouette:     {silhouette_score(X_casos,labels_km):.3f}")
print(f"  Davies-Bouldin: {davies_bouldin_score(X_casos,labels_km):.3f}")
print(f"  Calinski:       {calinski_harabasz_score(X_casos,labels_km):.1f}")

print("\nSILHOUETTE POR CLUSTER:")
for k in range(1, K_FINAL+1):
    mask=labels_km+1==k; s=sil_samples[mask]
    print(f"  C{k} (n={mask.sum()}): media={s.mean():.3f} | negativos={(s<0).sum()}")

aris_boot=[]
for _ in range(100):
    idx=resample(range(len(X_casos)),random_state=None)
    km_b=KMeans(n_clusters=K_FINAL,random_state=None,n_init=10)
    lbl_b=km_b.fit_predict(X_casos[idx])
    aris_boot.append(adjusted_rand_score(labels_km[idx],lbl_b))
print(f"\nESTABILIDAD BOOTSTRAP: ARI={np.mean(aris_boot):.3f} ± {np.std(aris_boot):.3f}")

print(f"\nCOMPARACION ALGORITMOS:")
for met,lbl in [('Spectral',labels_spec),('Jerárquico',labels_hc),('GMM',labels_gm)]:
    print(f"  {met:12s}: Sil={silhouette_score(X_casos,lbl):.3f} | "
          f"ARI_vs_KM={adjusted_rand_score(labels_km,lbl):.3f}")
if HDBSCAN_OK and n_cl_hdb>1:
    mask_v=labels_hdb!=-1
    print(f"  {'HDBSCAN':12s}: {n_cl_hdb} clusters, {n_noise_hdb} ruido, "
          f"Sil={silhouette_score(X_casos[mask_v],labels_hdb[mask_v]):.3f}")

pd.DataFrame({
    'metodo':         ['K-Means','Jerarquico','GMM','Spectral'],
    'silhouette':     [silhouette_score(X_casos,labels_km),
                       silhouette_score(X_casos,labels_hc),
                       silhouette_score(X_casos,labels_gm),
                       silhouette_score(X_casos,labels_spec)],
    'davies_bouldin': [davies_bouldin_score(X_casos,labels_km),
                       davies_bouldin_score(X_casos,labels_hc),
                       davies_bouldin_score(X_casos,labels_gm),
                       davies_bouldin_score(X_casos,labels_spec)],
    'ARI_vs_KMeans':  [1.0,ari_km_hc,ari_km_gm,ari_km_spec],
    'Bootstrap_ARI':  [np.mean(aris_boot),None,None,None],
}).to_csv(out('METRICAS_evaluacion.csv'),index=False,sep=';',encoding='utf-8-sig',decimal=',')
print("  ✓ METRICAS_evaluacion.csv")

#  17. MOSAICOS DE CROPS

try:
    from pylibCZIrw import czi as pyczi_cr
    CZI_OK_CR=True
except ImportError:
    pyczi_cr=None; CZI_OK_CR=False
    

try:
    from PIL import Image as PILImage, ImageDraw as PILDraw
    PIL_OK_CR=True
except ImportError:
    PIL_OK_CR=False

if CZI_OK_CR and PIL_OK_CR:
    print("\n"+"="*55+"\nGENERANDO MOSAICOS\n"+"="*55)

    def get_center_um_cr(czidoc):
        import xml.etree.ElementTree as _ET
        root=_ET.fromstring(czidoc.raw_metadata)
        for el in root.iter('CenterPosition'):
            if el.text:
                p=el.text.strip().split(',')
                if len(p)==2:
                    try: return float(p[0]),float(p[1])
                    except: pass
        return None,None

    def recortar_cr(czidoc,x1,y1,x2,y2):
        bbox=czidoc.total_bounding_box
        try: multi=len(czidoc.scenes_bounding_rectangle)>1
        except: multi=False
        def dentro(xs,ys,w,h):
            return (xs>=bbox['X'][0] and xs+w<=bbox['X'][1] and
                    ys>=bbox['Y'][0] and ys+h<=bbox['Y'][1])
        pxum=PIXEL_SIZE_UM; zoom=ZOOM_CZI
        if not multi:
            cx_um,cy_um=get_center_um_cr(czidoc)
            if cx_um is None: return None
            cx_px=(bbox['X'][0]+bbox['X'][1])/2; cy_px=(bbox['Y'][0]+bbox['Y'][1])/2
            px_l=int(round((x1*1000-cx_um)/pxum+cx_px)); px_r=int(round((x2*1000-cx_um)/pxum+cx_px))
            py_t=int(round(-(max(y1,y2)*1000-cy_um)/pxum+cy_px)); py_b=int(round(-(min(y1,y2)*1000-cy_um)/pxum+cy_px))
            xs=min(px_l,px_r); ys=min(py_t,py_b); w=abs(px_r-px_l); h=abs(py_b-py_t)
            if not dentro(xs,ys,w,h): return None
        else:
            tyb=bbox['Y'][1]
            xs=min(int(round(x1*1000/pxum)),int(round(x2*1000/pxum)))
            xe=max(int(round(x1*1000/pxum)),int(round(x2*1000/pxum)))
            yt=min(int(round(-(y1*1000/pxum)+tyb)),int(round(-(y2*1000/pxum)+tyb)))
            yb=max(int(round(-(y1*1000/pxum)+tyb)),int(round(-(y2*1000/pxum)+tyb)))
            w=xe-xs; h=yb-yt; ys=yt
            if not dentro(xs,ys,w,h):
                cx_um,cy_um=get_center_um_cr(czidoc)
                if cx_um is not None:
                    cx_px=(bbox['X'][0]+bbox['X'][1])/2; cy_px=(bbox['Y'][0]+bbox['Y'][1])/2
                    px_l=int(round((x1*1000-cx_um)/pxum+cx_px)); px_r=int(round((x2*1000-cx_um)/pxum+cx_px))
                    py_t=int(round(-(max(y1,y2)*1000-cy_um)/pxum+cy_px)); py_b=int(round(-(min(y1,y2)*1000-cy_um)/pxum+cy_px))
                    xs=min(px_l,px_r); ys=min(py_t,py_b); w=abs(px_r-px_l); h=abs(py_b-py_t)
                if not dentro(xs,ys,w,h): return None
        # reg=czidoc.read(roi=(xs,ys,w,h),zoom=zoom)
        # if reg is None or reg.size==0 or reg.max()==0: return None
        # img=PILImage.fromarray(reg[...,::-1].astype(np.uint8))
        # return img.resize(CROP_SIZE_PX,PILImage.LANCZOS)
        reg=czidoc.read(roi=(xs,ys,w,h),zoom=zoom)
        if reg is None or reg.size==0 or reg.max()==0: return None
        # Filtro: descartar si más del 10% de píxeles es negro o blanco
        img_arr = reg[..., :3]
        n_pixeles = img_arr.shape[0] * img_arr.shape[1]
        n_negro  = np.sum(img_arr.max(axis=2) < 15)
        n_blanco = np.sum(img_arr.min(axis=2) > 240)
        if (n_negro + n_blanco) / n_pixeles > 0.10:
            return None
        img=PILImage.fromarray(reg[...,::-1].astype(np.uint8))
        return img.resize(CROP_SIZE_PX,PILImage.LANCZOS)

    for cl in range(1, K_FINAL+1):
        col_hex=CLUSTER_COLORS[cl-1]
        rc=int(col_hex[1:3],16); gc=int(col_hex[3:5],16); bc=int(col_hex[5:7],16)
        casos_cl=df_comp_cluster[df_comp_cluster['cluster_kmeans']==cl]['Case'].tolist()
        if cl in medias_comp.index:
            comp_vals=medias_comp.loc[cl,pct_cols].values*100
            desc=f"(Tile C{np.argmax(comp_vals)+1} dom. {comp_vals.max():.0f}%)"
        else: desc=""
        print(f"\n  Cluster {cl} {desc} ({len(casos_cl)} casos)...")
        crops=[]; labels_crops=[]
        for caso in casos_cl:
            if len(crops)>=MAX_CROPS_TOTAL: break
            df_t=df[df['Case']==caso].sample(frac=1,random_state=42).reset_index(drop=True)
            if df_t.empty: continue
            czi_path=Path(CZI_FOLDER)/f"{caso}.czi"
            if not czi_path.exists(): print(f"    [sin CZI] {caso}"); continue
            n_caso=0
            try:
                with pyczi_cr.open_czi(str(czi_path)) as czidoc:
                    for _,tile_row in df_t.iterrows():
                        if len(crops)>=MAX_CROPS_TOTAL or n_caso>=MAX_CROPS_CASO: break
                        try:
                            img=recortar_cr(czidoc,
                                float(tile_row[col_x1]),float(tile_row[col_y1]),
                                float(tile_row[col_x2]),float(tile_row[col_y2]))
                            if img is not None:
                                crops.append(img); labels_crops.append(caso[-10:]); n_caso+=1
                        except: pass
            except Exception as e: print(f"    [ERROR {caso}] {e}")
        n_crops=len(crops); print(f"    → {n_crops} crops")
        if n_crops==0: continue
        W,H=CROP_SIZE_PX; ncols=min(10,n_crops); nrows=(n_crops+ncols-1)//ncols; gap=4; borde=5
        mosaic=PILImage.new('RGB',(ncols*W+(ncols-1)*gap,nrows*H+(nrows-1)*gap),(15,15,15))
        draw_m=PILDraw.Draw(mosaic)
        for k_idx,(img,lbl) in enumerate(zip(crops,labels_crops)):
            rm=k_idx//ncols; cm=k_idx%ncols; xo=cm*(W+gap); yo=rm*(H+gap)
            mosaic.paste(img,(xo,yo))
            draw_m.rectangle([xo,yo,xo+W-1,yo+H-1],outline=(rc,gc,bc),width=borde)
            draw_m.rectangle([xo,yo+H-15,xo+W,yo+H],fill=(0,0,0))
            draw_m.text((xo+4,yo+H-14),lbl,fill=(rc,gc,bc))
        fig,ax=plt.subplots(figsize=(ncols*2.4,nrows*2.4+1.0),facecolor='white')
        ax.imshow(np.array(mosaic)); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_edgecolor(col_hex); spine.set_linewidth(4)
        plt.title(f'Cluster {cl} — {n_crops} crops\n{desc}',
                  fontsize=13,fontweight='bold',color=col_hex,pad=8)
        plt.tight_layout()
        plt.savefig(out(f'MOSAICO_Cluster_{cl}.png'),dpi=180,bbox_inches='tight',facecolor='white')
        plt.close()
        

# RESUMEN FINAL

sil_km  = silhouette_score(X_casos, labels_km)
sil_sp  = silhouette_score(X_casos, labels_spec)
ari_sp  = adjusted_rand_score(labels_km, labels_spec)

print(f"""
{'='*60}
CLUSTERING POR COMPOSICION DE TILES COMPLETADO
{'='*60}
Carpeta: {OUTPUT_FOLDER}

PASO 1 — Tiles: k={K_TILES} tipos de tile
PASO 2 — Casos: k={K_FINAL} clusters

OUTLIERS: {n_outliers} → cluster=0: {outlier_casos if n_outliers>0 else 'Ninguno'}

DISTRIBUCION K-Means:
{dict(pd.Series(labels_km+1).value_counts().sort_index())}

METRICAS:
  K-Means  Sil={sil_km:.3f} | Bootstrap ARI={np.mean(aris_boot):.3f}±{np.std(aris_boot):.3f}
  Spectral Sil={sil_sp:.3f} | ARI vs KMeans={ari_sp:.3f}

¡
""")


# EXPORTAR PERFILES A CSV/EXCEL

# ── 1. Perfil de features biológicas por CLUSTER 
df_perfil_clusters = df_fn.copy()
df_perfil_clusters.index.name = 'feature'
df_perfil_clusters = df_perfil_clusters.reset_index()

df_perfil_clusters.to_csv(
    out('PERFIL_features_por_cluster.csv'),
    index=False, sep=';', encoding='utf-8-sig', decimal=','
)


# También en Excel con una hoja por cluster
with pd.ExcelWriter(out('PERFIL_features_por_cluster.xlsx'), engine='openpyxl') as writer:
    # Hoja completa con todos los clusters
    df_perfil_clusters.to_excel(writer, sheet_name='Todos_los_clusters', index=False)
    # Una hoja por cluster con ranking de features más altas
    for k in range(1, K_FINAL+1):
        cn = f'C{k}'
        if cn not in df_fn.columns:
            continue
        df_k = df_fn[[cn]].copy()
        df_k.columns = ['valor_normalizado_0_1']
        df_k.index.name = 'feature'
        df_k = df_k.reset_index().sort_values('valor_normalizado_0_1', ascending=False)
        n_casos_k = (df_comp_cluster['cluster_kmeans'] == k).sum()
        df_k.to_excel(writer, sheet_name=f'Cluster_{k}_n{n_casos_k}', index=False)


# ── 2. Perfil de features biológicas por TIPO DE TILE (equivalente a 00_perfil_cluster_tiles)

df_perfil_tiles = z_tiles.T.copy()  # features en filas, tiles en columnas
df_perfil_tiles.index.name = 'feature'
df_perfil_tiles.columns = [f'Tile_C{ct}' for ct in df_perfil_tiles.columns]
df_perfil_tiles = df_perfil_tiles.reset_index()

df_perfil_tiles.to_csv(
    out('PERFIL_features_por_tile.csv'),
    index=False, sep=';', encoding='utf-8-sig', decimal=','
)


# También en Excel con una hoja por tipo de tile
with pd.ExcelWriter(out('PERFIL_features_por_tile.xlsx'), engine='openpyxl') as writer:
    # Hoja completa
    df_perfil_tiles.to_excel(writer, sheet_name='Todos_los_tiles', index=False)
    # Una hoja por tipo de tile con ranking por z-score absoluto
    for ct in range(1, K_TILES+1):
        col_tile = f'Tile_C{ct}'
        if col_tile not in df_perfil_tiles.columns:
            continue
        df_ct = df_perfil_tiles[['feature', col_tile]].copy()
        df_ct.columns = ['feature', 'z_score']
        df_ct['abs_z_score'] = df_ct['z_score'].abs()
        df_ct = df_ct.sort_values('abs_z_score', ascending=False).drop(columns='abs_z_score')
        n_tiles_ct = (labels_tiles + 1 == ct).sum()
        df_ct.to_excel(writer, sheet_name=f'Tile_C{ct}_n{n_tiles_ct}', index=False)
print("  ✓ PERFIL_features_por_tile.xlsx")


# py -3.12 c:/Users/carme/OneDrive/Escritorio/M.UCM/TFM/1700MICRAS/CENTROIDE/FEATURE_ENGINEERING/clustering_features_eda2.py