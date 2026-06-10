import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import numpy as np

#  CARGA DE DATOS 
INPUT_FILE = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\image_based_metrics_objects.tsv"
OUTPUT_PLOT1 = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\QC\cobertura_area_analizada.png"
OUTPUT_PLOT2 = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\QC\composicion_tisular.png"
OUTPUT_PLOT3 = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\QC\intensidades_medias.png"
OUTPUT_PLOT4 = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\QC\intensidades.png"



df = pd.read_csv(INPUT_FILE, sep='\t')

#  AGREGACIÓN POR PACIENTE 
#
areas = [
    'Area (um2) (Analyzed ROI)',
    'Area (um2) (Pathological ROI)',
    'Area (um2) (Tumor in Analyzed ROI)',
    'Area (um2) (Muscle in Analyzed ROI)'
]
pat = df.groupby('Name')[areas].sum().reset_index()

# Excluir pacientes sin área analizada
pat = pat[pat['Area (um2) (Analyzed ROI)'] > 0].copy()

# Convertir de µm² a mm²  los ejes
for col in areas:
    pat[col] = pat[col] / 1e6

pat = pat.sort_values('Name').reset_index(drop=True)
patients = pat['Name'].tolist()
x = np.arange(len(patients))


# PLOT 1: Cobertura de área analizada

fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(x, pat['Area (um2) (Pathological ROI)'],
        '-o', color='#C0392B', linewidth=1.8, markersize=4,
        label='Área patológica')

ax.plot(x, pat['Area (um2) (Analyzed ROI)'],
        '--s', color='#2980B9', linewidth=1.8, markersize=4,
        label='Área analizada')

ax.set_xticks(x)
ax.set_xticklabels(patients, rotation=90, fontsize=7)
ax.set_ylabel('Área (mm²)', fontsize=11)
ax.set_xlabel('Paciente', fontsize=11)
ax.set_title('Cobertura de área analizada', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Guardado: {OUTPUT_PLOT1}")


#PLOT 2: Composición tisular del área analizada

# Calcular área "no tumor / no músculo" y porcentajes
pat['Area_NoTumorNoMuscle'] = (
    pat['Area (um2) (Analyzed ROI)']
    - pat['Area (um2) (Tumor in Analyzed ROI)']
    - pat['Area (um2) (Muscle in Analyzed ROI)']
).clip(lower=0)

pat['pct_tumor']  = pat['Area (um2) (Tumor in Analyzed ROI)']  / pat['Area (um2) (Analyzed ROI)'] * 100
pat['pct_muscle'] = pat['Area (um2) (Muscle in Analyzed ROI)'] / pat['Area (um2) (Analyzed ROI)'] * 100
pat['pct_other']  = pat['Area_NoTumorNoMuscle']                / pat['Area (um2) (Analyzed ROI)'] * 100

fig, ax = plt.subplots(figsize=(16, 5))

ax.bar(x, pat['pct_tumor'],
       color='#C0392B', label='% Tumor')
ax.bar(x, pat['pct_muscle'],
       bottom=pat['pct_tumor'],
       color='#2980B9', label='% Músculo')
ax.bar(x, pat['pct_other'],
       bottom=pat['pct_tumor'] + pat['pct_muscle'],
       color='#95A5A6', label='% No tumor / No músculo')

ax.set_xticks(x)
ax.set_xticklabels(patients, rotation=90, fontsize=7)
ax.set_ylabel('%', fontsize=11)
ax.set_xlabel('Paciente', fontsize=11)
ax.set_title('Composición tisular del área analizada', fontsize=13, fontweight='bold')
ax.set_ylim(0, 100)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Guardado: {OUTPUT_PLOT2}")


#  PLOT 3: Intensidades medias (CK en Tumor, Mag en Músculo, Hx en Analyzed ROI)

intensity_cols = [
    'Mean Intensity CK (Tumor in Analyzed ROI)',
    'Mean Intensity Mag (Muscle in Analyzed ROI)',
    'Mean Intensity Hematoxilin (Analyzed ROI)'
]

# Para intensidad usamos la media ponderada por número de filas por paciente

int_pat = df.groupby('Name')[intensity_cols].mean().reset_index()

# Filtramos solo pacientes con área analizada > 0 
valid_patients = pat['Name'].tolist()
int_pat = int_pat[int_pat['Name'].isin(valid_patients)].copy()
int_pat = int_pat.sort_values('Name').reset_index(drop=True)

x2 = np.arange(len(int_pat))

fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
fig.suptitle('Intensidades medias por marcador', fontsize=14, fontweight='bold', y=1.01)

configs = [
    ('Mean Intensity CK (Tumor in Analyzed ROI)',        '#C0392B', 'CK — Mean Intensity (Tumor in Analyzed ROI)'),
    ('Mean Intensity Mag (Muscle in Analyzed ROI)',       '#8E44AD', 'Magenta — Mean Intensity (Muscle in Analyzed ROI)'),
    ('Mean Intensity Hematoxilin (Analyzed ROI)',         '#2980B9', 'Hematoxilina — Mean Intensity (Analyzed ROI)'),
]

for ax, (col, color, title) in zip(axes, configs):
    ax.plot(x2, int_pat[col], '-o', color=color, linewidth=1.8, markersize=4)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('Intensidad media', fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

axes[-1].set_xticks(x2)
axes[-1].set_xticklabels(int_pat['Name'].tolist(), rotation=90, fontsize=7)
axes[-1].set_xlabel('Paciente', fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT3, dpi=150, bbox_inches='tight')
plt.close()
print(f"Guardado: {OUTPUT_PLOT3}")


# ¡ PLOT 3: Intensidades medias

intensity_cols = [
    'Mean Intensity CK (Tumor in Analyzed ROI)',
    'Mean Intensity CK (Analyzed ROI)',
    'Mean Intensity Mag (Muscle in Analyzed ROI)',
    'Mean Intensity Mag (Analyzed ROI)',
    'Mean Intensity Hematoxilin (Analyzed ROI)',
    'Mean Intensity Hematoxilin (No Tumor/No Muscle in Analyzed ROI)'
]

int_pat = df.groupby('Name')[intensity_cols].mean().reset_index()
valid_patients = pat['Name'].tolist()
int_pat = int_pat[int_pat['Name'].isin(valid_patients)].copy()
int_pat = int_pat.sort_values('Name').reset_index(drop=True)

x2 = np.arange(len(int_pat))

fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
fig.suptitle('Intensidades medias por marcador', fontsize=14, fontweight='bold')

# CK
axes[0].plot(x2, int_pat['Mean Intensity CK (Tumor in Analyzed ROI)'],
             '-o', color='#C0392B', linewidth=1.8, markersize=4, label='CK en Tumor')
axes[0].plot(x2, int_pat['Mean Intensity CK (Analyzed ROI)'],
             '--o', color='#C0392B', linewidth=1.8, markersize=4, alpha=0.5, label='CK en Analyzed ROI')
axes[0].set_title('CK', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Intensidad media', fontsize=10)
axes[0].legend(fontsize=9)
axes[0].yaxis.grid(True, linestyle='--', alpha=0.5)
axes[0].set_axisbelow(True)

# Magenta
axes[1].plot(x2, int_pat['Mean Intensity Mag (Muscle in Analyzed ROI)'],
             '-o', color='#8E44AD', linewidth=1.8, markersize=4, label='Mag en Músculo')
axes[1].plot(x2, int_pat['Mean Intensity Mag (Analyzed ROI)'],
             '--o', color='#8E44AD', linewidth=1.8, markersize=4, alpha=0.5, label='Mag en Analyzed ROI')
axes[1].set_title('Magenta', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Intensidad media', fontsize=10)
axes[1].legend(fontsize=9)
axes[1].yaxis.grid(True, linestyle='--', alpha=0.5)
axes[1].set_axisbelow(True)

# Hematoxilina
axes[2].plot(x2, int_pat['Mean Intensity Hematoxilin (Analyzed ROI)'],
             '-o', color='#2980B9', linewidth=1.8, markersize=4, label='Hx en Analyzed ROI')
axes[2].plot(x2, int_pat['Mean Intensity Hematoxilin (No Tumor/No Muscle in Analyzed ROI)'],
             '--o', color='#2980B9', linewidth=1.8, markersize=4, alpha=0.5, label='Hx en No Tumor/No Músculo')
axes[2].set_title('Hematoxilina', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Intensidad media', fontsize=10)
axes[2].legend(fontsize=9)
axes[2].yaxis.grid(True, linestyle='--', alpha=0.5)
axes[2].set_axisbelow(True)

axes[-1].set_xticks(x2)
axes[-1].set_xticklabels(int_pat['Name'].tolist(), rotation=90, fontsize=7)
axes[-1].set_xlabel('Paciente', fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT3, dpi=150, bbox_inches='tight')
plt.close()
print(f"Guardado: {OUTPUT_PLOT4}")