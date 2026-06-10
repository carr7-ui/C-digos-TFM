
# CONFIGURACIÓN


OUTPUT_FOLDER = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\1700MICRAS\CENTROIDE\FEATURE_ENGINEERING\FINAL"
CZI_FOLDER    = r"\\imgserver\IMAGES\CONFOCAL\IA\crodriguezj\images"

PIXEL_SIZE_UM = 0.1723
ZOOM_CZI      = 0.15
N_BINS        = 10
CELDA_PX      = 160   # tamaño de cada celda en el grid final

#  IMPORTS

import os, warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

try:
    from pylibCZIrw import czi as pyczi
except ImportError:
    raise ImportError("pylibCZIrw no está instalado.\n"
                      )

warnings.filterwarnings('ignore')

csv_guia = os.path.join(OUTPUT_FOLDER, "GUIA_CENTROIDE.csv")
if not os.path.exists(csv_guia):
    raise FileNotFoundError(f"No se encuentra {csv_guia}\nEjecuta A1_EDA.py .")

tiles_dir = Path(OUTPUT_FOLDER) / "tiles_png"
tiles_dir.mkdir(parents=True, exist_ok=True)

def out(f): return os.path.join(OUTPUT_FOLDER, f)

# FUNCIÓN DE RECORTE CZI

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
            px_l = int(round((x1*1000-cx2)/pxum+(bbox['X'][0]+bbox['X'][1])/2))
            px_r = int(round((x2*1000-cx2)/pxum+(bbox['X'][0]+bbox['X'][1])/2))
            py_t = int(round(-(max(y1,y2)*1000-cy2)/pxum+(bbox['Y'][0]+bbox['Y'][1])/2))
            py_b = int(round(-(min(y1,y2)*1000-cy2)/pxum+(bbox['Y'][0]+bbox['Y'][1])/2))
            xs = min(px_l,px_r); ys = min(py_t,py_b)
            w  = abs(px_r-px_l); h  = abs(py_b-py_t)
            if not dentro(xs, ys, w, h): return None

    else:
        # Intento 1: Y invertido respecto al borde inferior del bbox total
        total_y_bottom = bbox['Y'][1]
        xs = min(int(round(x1*1000/pxum)), int(round(x2*1000/pxum)))
        ys = min(int(round(-(y1*1000/pxum)+total_y_bottom)),
                 int(round(-(y2*1000/pxum)+total_y_bottom)))
        w  = abs(int(round(x2*1000/pxum)) - int(round(x1*1000/pxum)))
        h  = abs(int(round(-(y2*1000/pxum)+total_y_bottom)) -
                 int(round(-(y1*1000/pxum)+total_y_bottom)))

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
                if not dentro(xs, ys, w, h): return None

    # reg = czidoc.read(roi=(xs, ys, w, h), zoom=zoom)
    # if reg is None or reg.size == 0 or reg.max() == 0: return None
    # img = Image.fromarray(reg[..., ::-1].astype(np.uint8))
    # return img.resize((int(round(w*zoom)), int(round(h*zoom))), Image.LANCZOS)
    reg = czidoc.read(roi=(xs, ys, w, h), zoom=zoom)
    if reg is None or reg.size == 0 or reg.max() == 0: return None

# Filtro
    img_arr = reg[..., :3]
    n_pixeles = img_arr.shape[0] * img_arr.shape[1]
    n_negro  = np.sum(img_arr.max(axis=2) < 15)
    n_blanco = np.sum(img_arr.min(axis=2) > 240)
    if (n_negro + n_blanco) / n_pixeles > 0.10:
        return None

    img = Image.fromarray(reg[..., ::-1].astype(np.uint8))
    return img.resize((int(round(w*zoom)), int(round(h*zoom))), Image.LANCZOS)
# BUCLE PRINCIPAL

df_guia = pd.read_csv(csv_guia, sep=';', encoding='utf-8-sig')
df_guia = df_guia.rename(columns={
    'x1_Izquierda': 'env_x1', 'y1_Arriba': 'env_y1',
    'x2_Derecha':   'env_x2', 'y2_Abajo':  'env_y2',
})

grid_lookup  = {}   # (bin_col, bin_row) 
tiles_ok     = 0
tiles_error  = 0

casos_totales = sorted(df_guia['Case'].unique())
print(f"Casos a procesar: {len(casos_totales)}\n")

for case, grupo in df_guia.groupby('Case'):
    czi_path = Path(CZI_FOLDER) / f"{case}.czi"
    if not czi_path.exists():
        print(f"[AVISO] CZI no encontrado: {case}")
        continue

    case_dir = tiles_dir / case
    case_dir.mkdir(exist_ok=True)

    # Detectar tiles ya recortados para este caso (reanudación automática)
    ya_recortados = {p.stem.split('_')[0] + '_' + p.stem.split('_')[1]  # bin_col_row
                     for p in case_dir.glob("*.png")}

    tiles_pendientes = [
        row for _, row in grupo.iterrows()
        if f"bin_{int(row['bin_col'])}_{int(row['bin_row'])}" not in ya_recortados
    ]

    if not tiles_pendientes:
        print(f"{case}: todos los tiles ya recortados ({len(grupo)} bins), cargando para grid...")
        # Cargar los ya existentes para el grid global
        for _, row in grupo.iterrows():
            bc = int(row['bin_col']); br = int(row['bin_row'])
            patron = list(case_dir.glob(f"bin_{bc}_{br}_*.png"))
            if patron and (bc, br) not in grid_lookup:
                grid_lookup[(bc, br)] = Image.open(patron[0])
        continue

    print(f"\n{'─'*50}")
    print(f"Caso: {case}  ({len(tiles_pendientes)} tiles pendientes de {len(grupo)} total)")
    print(f"Carpeta: {case_dir}")
    print(f"{'─'*50}")

    try:
        with pyczi.open_czi(str(czi_path)) as czidoc:
            for row in tiles_pendientes:
                bc  = int(row['bin_col'])
                br  = int(row['bin_row'])
                bid = row['ID_Cuadrado_PCA']

                print(f"  → {bid} (bin {bc},{br})... ", end='', flush=True)

                try:
                    img = recortar_tile_czi(
                        czidoc,
                        row['env_x1'], row['env_y1'],
                        row['env_x2'], row['env_y2'],
                        pxum=PIXEL_SIZE_UM, zoom=ZOOM_CZI
                    )
                except Exception as e:
                    print(f"ERROR: {e}")
                    tiles_error += 1
                    continue

                if img is None:
                    print("VACÍO (región fuera del bbox)")
                    tiles_error += 1
                    continue

                # Guardar PNG individual en la carpeta del caso
                png_path = case_dir / f"{bid}_{case}.png"
                img.save(png_path)
                tiles_ok += 1

                w_px, h_px = img.size
                print(f"OK ({w_px}×{h_px}px guardado)")

                # Registrar para el grid global (primer tile que llegue a ese bin)
                if (bc, br) not in grid_lookup:
                    grid_lookup[(bc, br)] = img

    except Exception as e:
        print(f"  [ERROR abriendo CZI] {e}")
        continue

    print(f"  Caso completado. Tiles OK hasta ahora: {tiles_ok}")

#  GRID 10×10 DE THUMBNAILS

print(f"\nEnsamblando grid {N_BINS}×{N_BINS}... ({len(grid_lookup)} celdas con imagen)")

gs = N_BINS * CELDA_PX
grid_img = Image.new('RGB', (gs, gs), (40, 40, 40))   # fondo oscuro

for (bc, br), tile_img in grid_lookup.items():
    thumb = tile_img.resize((CELDA_PX, CELDA_PX), Image.LANCZOS)
    xoff  = bc * CELDA_PX
    yoff  = (N_BINS - 1 - br) * CELDA_PX   # Y invertido: bin 0 abajo, bin 9 arriba
    grid_img.paste(thumb, (xoff, yoff))

# Líneas de la cuadrícula
draw = ImageDraw.Draw(grid_img)
for i in range(N_BINS + 1):
    draw.line([(i*CELDA_PX, 0), (i*CELDA_PX, gs)], fill=(180, 180, 180), width=1)
    draw.line([(0, i*CELDA_PX), (gs, i*CELDA_PX)], fill=(180, 180, 180), width=1)

grid_path = out("08_grid_thumbnails_PCA.png")
grid_img.save(grid_path)


print(f"""
{'='*50}
CROPS CZI COMPLETADOS
{'='*50}
Tiles recortados OK : {tiles_ok}
Tiles con error     : {tiles_error}
Celdas en el grid   : {len(grid_lookup)} / {N_BINS*N_BINS}


""")

#py -3.12 c:/Users/carme/OneDrive/Escritorio/M.UCM/TFM/1700MICRAS/CENTROIDE/FEATURE_ENGINEERING/new_eda2/A3_EDA.py
