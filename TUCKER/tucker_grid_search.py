"""

Grid search para seleccionar el número óptimo de patrones (N_PAT)
en la descomposición Tucker no-negativa del atlas de vejiga.


"""

# CONFIGURACIÓN

DF_ALL_CSV    = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\1700MICRAS\CENTROIDE\FEATURE_ENGINEERING\new_eda2\df_all_completo.csv"
OUTPUT_FOLDER = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\1700MICRAS\CENTROIDE\FEATURE_ENGINEERING\TUCKER\GRID_SEARCH"

N_BINS   = 10
RP_RANGE = [3, 4, 5, 6, 7]    # número de patrones de paciente a explorar
RS_RANGE = [3, 4, 5]           # número de patrones espaciales a explorar
N_SEEDS  = 3                  # seeds por combinación (para estabilidad)
N_PAT_ELEGIDO = 5              

#  IMPORTS

import os, warnings, time
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorly as tl
from tensorly.decomposition import non_negative_tucker

warnings.filterwarnings('ignore')
tl.set_backend('numpy')
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
def out(f): return os.path.join(OUTPUT_FOLDER, f)

#  1. CONSTRUIR TENSOR

print("Cargando datos y construyendo tensor...")
df = pd.read_csv(DF_ALL_CSV)
print(f"  Tiles: {len(df):,} | Casos: {df['Case'].nunique()}")

pc1_edges = np.linspace(df['PC1'].min(), df['PC1'].max(), N_BINS+1)
pc2_edges = np.linspace(df['PC2'].min(), df['PC2'].max(), N_BINS+1)
df['_b1'] = pd.cut(df['PC1'], bins=pc1_edges, labels=False, include_lowest=True).astype(int)
df['_b2'] = pd.cut(df['PC2'], bins=pc2_edges, labels=False, include_lowest=True).astype(int)

casos   = sorted(df['Case'].unique())
N_CASOS = len(casos)
c_idx   = {c:i for i,c in enumerate(casos)}

tensor = np.zeros((N_CASOS, N_BINS, N_BINS))
for caso, grupo in df.groupby('Case'):
    i = c_idx[caso]; total = len(grupo)
    for (b1,b2), cnt in grupo.groupby(['_b1','_b2']).size().items():
        tensor[i, int(b1), int(b2)] = cnt / total

norm_tensor = np.linalg.norm(tensor)
print(f"  Tensor: {tensor.shape} | norma={norm_tensor:.4f}")

#  2. GRID SEARCH

print(f"\nGrid search: rp={RP_RANGE} x rs={RS_RANGE} x {N_SEEDS} seeds")
print(f"Total combinaciones: {len(RP_RANGE)*len(RS_RANGE)} x {N_SEEDS} seeds = "
      f"{len(RP_RANGE)*len(RS_RANGE)*N_SEEDS} ejecuciones Tucker\n")

resultados = []
total = len(RP_RANGE) * len(RS_RANGE)
n = 0

for rp, rs in product(RP_RANGE, RS_RANGE):
    n += 1
    t0 = time.time()
    errores = []

    for seed in range(N_SEEDS):
        core, facs = non_negative_tucker(
            tensor, rank=[rp, rs, rs],
            n_iter_max=50, random_state=seed, verbose=False
        )
        r   = tl.tucker_to_tensor((core, facs))
        err = np.linalg.norm(tensor - r) / norm_tensor
        errores.append(err)

    err_medio = np.mean(errores)
    err_std   = np.std(errores)
    err_min   = np.min(errores)
    elapsed   = time.time() - t0

    resultados.append({
        'rp': rp, 'rs': rs,
        'error_medio': err_medio,
        'error_std':   err_std,
        'error_min':   err_min,
        'tiempo_s':    elapsed
    })

    print(f"  [{n:2d}/{total}] rp={rp} rs={rs} | "
          f"error={err_medio:.4f}±{err_std:.4f} | {elapsed:.1f}s")

df_res = pd.DataFrame(resultados)
df_res.to_csv(out('GRID_SEARCH_resultados.csv'), index=False, sep=';', encoding='utf-8-sig')


#3. FIGURA PRINCIPAL: CURVA ERROR VS rp

COLORS_RS = {3: '#E74C3C', 4: '#3498DB', 5: '#2ECC71'}

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
fig.suptitle('Grid Search Tucker No-Negativo — Selección del Número de Patrones\n'
             f'Atlas de Vejiga | {N_CASOS} casos | Tensor {N_BINS}×{N_BINS}',
             fontsize=13, fontweight='bold')

# Panel izquierdo: curvas por rs
ax = axes[0]
for rs in RS_RANGE:
    sub = df_res[df_res['rs']==rs].sort_values('rp')
    ax.plot(sub['rp'], sub['error_medio'],
            'o-', color=COLORS_RS[rs], lw=2, ms=8,
            label=f'rs={rs}')
    ax.fill_between(sub['rp'],
                    sub['error_medio'] - sub['error_std'],
                    sub['error_medio'] + sub['error_std'],
                    color=COLORS_RS[rs], alpha=0.12)

# Marcar el rp elegido
for rs in RS_RANGE:
    sub = df_res[(df_res['rs']==rs) & (df_res['rp']==N_PAT_ELEGIDO)]
    if not sub.empty:
        ax.scatter(N_PAT_ELEGIDO, sub['error_medio'].values[0],
                   color=COLORS_RS[rs], s=180, zorder=5,
                   marker='*', edgecolors='black', lw=1.2)

ax.axvline(N_PAT_ELEGIDO, color='black', linestyle='--', lw=1.8, alpha=0.7,
           label=f'rp elegido = {N_PAT_ELEGIDO} (codo)')
ax.set_xlabel('Número de patrones de paciente (rp)', fontsize=11)
ax.set_ylabel('Error de reconstrucción relativo', fontsize=11)
ax.set_title('Curva error vs rp\n(banda = ±1 SD entre seeds)',
             fontsize=11, fontweight='bold')
ax.set_xticks(RP_RANGE)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(alpha=0.3)

# Añadir anotación del codo
sub_elec = df_res[(df_res['rp']==N_PAT_ELEGIDO)].groupby('rp')['error_medio'].mean()
if not sub_elec.empty:
    ax.annotate(f'Codo en rp={N_PAT_ELEGIDO}\nerror≈{sub_elec.values[0]:.3f}',
                xy=(N_PAT_ELEGIDO, sub_elec.values[0]),
                xytext=(N_PAT_ELEGIDO+0.3, sub_elec.values[0]+0.015),
                fontsize=9, fontweight='bold', color='black',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))

# Panel derecho: reducción marginal del error
ax2 = axes[1]
for rs in RS_RANGE:
    sub = df_res[df_res['rs']==rs].sort_values('rp')
    errs = sub['error_medio'].values
    rps  = sub['rp'].values
    # Diferencia entre rp consecutivos (ganancia marginal)
    delta = np.abs(np.diff(errs))
    rps_mid = [(rps[i]+rps[i+1])/2 for i in range(len(rps)-1)]
    ax2.plot(rps_mid, delta, 'o-', color=COLORS_RS[rs], lw=2, ms=8,
             label=f'rs={rs}')

ax2.axvline(N_PAT_ELEGIDO - 0.5, color='black', linestyle='--', lw=1.8, alpha=0.7,
            label=f'rp={N_PAT_ELEGIDO} (ganancia marginal ↓)')
ax2.set_xlabel('rp (punto medio entre valores consecutivos)', fontsize=11)
ax2.set_ylabel('Reducción marginal del error |Δerror|', fontsize=11)
ax2.set_title('Ganancia marginal al añadir un patrón más\n'
              '(el codo coincide con la caída brusca)',
              fontsize=11, fontweight='bold')
ax2.legend(fontsize=10, framealpha=0.9)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(out('GRID_SEARCH_curva_error.png'), dpi=220,
            bbox_inches='tight', facecolor='white')
plt.close()


#  4. HEATMAP: ERROR POR COMBINACIÓN rp x rs

pivot = df_res.pivot(index='rs', columns='rp', values='error_medio')

fig, ax = plt.subplots(figsize=(9, 5), facecolor='white')
sns.heatmap(pivot, ax=ax, cmap='YlOrRd_r', annot=True, fmt='.4f',
            cbar_kws={'label': 'Error reconstrucción medio', 'shrink': 0.8},
            linewidths=0.5, linecolor='white')

# Resaltar la celda elegida
col_idx = list(pivot.columns).index(N_PAT_ELEGIDO)
for rs_idx, rs in enumerate(sorted(RS_RANGE)):
    ax.add_patch(plt.Rectangle((col_idx, rs_idx), 1, 1,
                                fill=False, edgecolor='black', lw=3, zorder=5))

ax.set_xlabel('Número de patrones de paciente (rp)', fontsize=11)
ax.set_ylabel('Número de patrones espaciales (rs)', fontsize=11)
ax.set_title(f'Error de reconstrucción Tucker — Grid Search rp×rs\n'
             f'Recuadro negro = configuración elegida rp={N_PAT_ELEGIDO}, rs=5',
             fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(out('GRID_SEARCH_heatmap_error.png'), dpi=220,
            bbox_inches='tight', facecolor='white')
plt.close()

# 5. RESUMEN


mejor = df_res.loc[df_res['error_medio'].idxmin()]
elec  = df_res[(df_res['rp']==N_PAT_ELEGIDO) & (df_res['rs']==5)]




#py -3.12 c:/Users/carme/Downloads/tucker_grid_search.py