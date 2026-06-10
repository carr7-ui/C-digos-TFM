
#  CONFIGURACIÓN

OUTPUT_FOLDER = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\1700MICRAS\CENTROIDE\FEATURE_ENGINEERING\new_eda2"
MLD_FOLDER    = r"\\imgserver\IMAGES\CONFOCAL\IA\crodriguezj\annotations\1700_microns"

TOLERANCIA_FRACCION = 0.6   # fracción del tamaño del tile para el match

#  IMPORTS

import os, re, struct, warnings
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon

warnings.filterwarnings('ignore')

csv_guia = os.path.join(OUTPUT_FOLDER, "GUIA_CENTROIDE.csv")
if not os.path.exists(csv_guia):
    raise FileNotFoundError(f"No se encuentra {csv_guia}\nEjecuta A1_pca_y_guia.py primero.")

Path(OUTPUT_FOLDER, "solapamiento_mld").mkdir(parents=True, exist_ok=True)
def out(f): return os.path.join(OUTPUT_FOLDER, f)


# DEFINICIONES DE COLOR 
COLOR_CAT = {
    "tumor":  ('#E74C3C', 0.45, "Tumor"),
    "stroma": ("#34DB50", 0.45, "Estroma/Fibroso"),
    "muscle": ("#361EE9", 0.45, "Músculo"),
    "bk":     (None,      0.0,  "Background"),
    "otros":  ('#FF00FF', 0.40, "Otros"),
}
LABEL_NUM = {1:"stroma", 2:"tumor", 3:"bk", 4:"stroma",
             5:"muscle", 6:"stroma", 7:"stroma", 8:"stroma", 9:"stroma"}
TID_DEF   = {0:"stroma", 1:"stroma", 2:"tumor", 3:"bk", 4:"muscle", 5:"muscle"}


# FUNCIONES MLD

def normalizar_cat(nombre):
    if not nombre: return None
    n = nombre.lower()
    if any(x in n for x in ['tumor', 'tumour']): return "tumor"
    if any(x in n for x in ['stroma', 'fibrous']): return "stroma"
    if any(x in n for x in ['muscle', 'muscul']): return "muscle"
    if any(x in n for x in ['bk', 'background']): return "bk"
    if any(x in n for x in ['vessel', 'vasc']): return "stroma"
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
        with open(path, "rb") as f:
            content = f.read()
        tag = b"[LayerConfigs]"
        idx = content.find(tag)
        if idx == -1: return res
        ptr = idx + len(tag) + 1
        length = struct.unpack("<Q", content[ptr:ptr+8])[0]
        xml_s = content[ptr+8:ptr+8+length].decode("utf-8", errors="ignore")
        root = ET.fromstring(xml_s)
        for ln in root.findall("Layer"):
            lname = ln.get("Name", "")
            res[lname] = {}
            for tn in ln.findall("Type"):
                ia = tn.get("Index"); nn = tn.find("n")
                if ia and nn is not None and nn.text:
                    res[lname][int(ia)] = nn.text.strip()
    except Exception as e:
        print(f"  [LayerConfigs] {e}")
    return res


def leer_mld(path):
    objetos = []
    with open(path, "rb") as f:
        assert f.read(4).decode("ascii", "ignore") == "LDFF"
        f.read(4)
        n_layers = struct.unpack("<i", f.read(4))[0]
        for _ in range(n_layers):
            hdr = f.read(69)
            if len(hdr) < 69: break
            lname = hdr[:64].split(b'\x00')[0].decode("ascii", "ignore").strip()
            n_obj = struct.unpack("<i", hdr[65:69])[0]
            read = 0
            while read < n_obj:
                sr = f.read(4)
                if not sr: break
                buf = f.read(struct.unpack("<i", sr)[0])
                off = 0
                while off < len(buf) and read < n_obj:
                    try:
                        shape = buf[off]; off += 1
                        ttype = buf[off]; off += 1
                        pts = []; bbox = None
                        if shape in [0, 8]:
                            np_ = struct.unpack("<i", buf[off:off+4])[0]; off += 4
                            for _ in range(np_):
                                x = struct.unpack("<f", buf[off:off+4])[0]
                                y = struct.unpack("<f", buf[off+4:off+8])[0]
                                pts.append((x, y)); off += 8
                            if pts:
                                xs, ys = zip(*pts)
                                bbox = (min(xs), min(ys), max(xs), max(ys))
                        elif shape == 5:
                            off += 4
                            cx = struct.unpack("<d", buf[off:off+8])[0]; off += 8
                            cy = struct.unpack("<d", buf[off:off+8])[0]; off += 8
                            w  = struct.unpack("<d", buf[off:off+8])[0]; off += 8
                            h  = struct.unpack("<d", buf[off:off+8])[0]; off += 8
                            _  = struct.unpack("<d", buf[off:off+8])[0]; off += 8
                            if w < 1e30 and h < 1e30:
                                bbox = (cx-w/2, cy-h/2, cx+w/2, cy+h/2)
                                pts = [(cx-w/2,cy-h/2),(cx+w/2,cy-h/2),
                                       (cx+w/2,cy+h/2),(cx-w/2,cy+h/2)]
                        elif shape == 6:
                            off += 4
                            cx = struct.unpack("<d", buf[off:off+8])[0]; off += 8
                            cy = struct.unpack("<d", buf[off:off+8])[0]; off += 8
                            wh = struct.unpack("<d", buf[off:off+8])[0]; off += 8
                            _  = struct.unpack("<d", buf[off:off+8])[0]; off += 8
                            if wh < 1e30:
                                bbox = (cx-wh/2, cy-wh/2, cx+wh/2, cy+wh/2)
                                pts = [(cx-wh/2,cy-wh/2),(cx+wh/2,cy-wh/2),
                                       (cx+wh/2,cy+wh/2),(cx-wh/2,cy+wh/2)]
                        elif shape == 1: off += 4+16+24
                        elif shape == 2: off += 4+16+8
                        elif shape == 3:
                            np2 = struct.unpack("<i", buf[off:off+4])[0]; off += 4
                            off += np2 * 8
                        elif shape in [4, 9]: off += 16
                        elif shape == 7: off += 20
                        for _ in range(2):
                            while off < len(buf) and buf[off] != 0: off += 1
                            off += 1
                        if bbox is not None:
                            objetos.append({
                                'layer': lname, 'type_id': ttype,
                                'points': pts, 'bbox': bbox,
                                'w': abs(bbox[2]-bbox[0]),
                                'h': abs(bbox[3]-bbox[1]),
                            })
                        read += 1
                    except Exception: break
    return objetos


def separar_mld(objetos, nombres_por_capa):
    roi_cand = [o for o in objetos
        if o['layer'] == 'ROI' and 0.2 < o['w'] < 2.5 and abs(o['w']-o['h']) < 0.2]
    tid = Counter(o['type_id'] for o in roi_cand).most_common(1)[0][0] if roi_cand else None
    tiles = [o for o in objetos
     if o['layer'] == 'ROI' and o['type_id'] == tid and o['w'] < 3.0]
    nl = nombres_por_capa.get('Label', {})
    anots = []
    for o in objetos:
        if o['layer'] != 'Label' or o['w'] > 5 or o['h'] > 5: continue
        cat = categoria_obj(o, nl)
        if cat == 'bk': continue
        o['categoria'] = cat
        anots.append(o)
    ts = float(np.median([t['w'] for t in tiles])) if tiles else 0.353
    print(f"  Tiles ROI: {len(tiles)} ({ts:.4f}mm) | Anotaciones: {len(anots)}")
    return tiles, anots, ts


# BUCLE PRINCIPAL

df_guia = pd.read_csv(csv_guia, sep=';', encoding='utf-8-sig')
df_guia = df_guia.rename(columns={
    'x1_Izquierda': 'env_x1', 'y1_Arriba': 'env_y1',
    'x2_Derecha':   'env_x2', 'y2_Abajo':  'env_y2',
})

casos_totales = sorted(df_guia['Case'].unique())
ya_hechos = {p.stem.replace('_solapamiento', '')
             for p in Path(out("solapamiento_mld")).glob('*_solapamiento.png')}
pendientes = [c for c in casos_totales if c not in ya_hechos]

print(f"Casos totales : {len(casos_totales)}")
print(f"Ya procesados : {len(ya_hechos)}")
print(f"Pendientes    : {len(pendientes)}\n")

procesados = 0
errores = 0

for case in pendientes:
    grupo    = df_guia[df_guia['Case'] == case]
    mld_path = Path(MLD_FOLDER) / f"{case}.mld"

    if not mld_path.exists():
        print(f"[AVISO] MLD no encontrado: {case}")
        errores += 1
        continue

    print(f"\n=== {case} ({len(grupo)} tiles) ===")
    nombres = leer_layer_configs(str(mld_path))

    try:
        objetos = leer_mld(str(mld_path))
    except Exception as e:
        print(f"  [ERROR MLD] {e}")
        errores += 1
        continue

    tiles_mld, anots, ts = separar_mld(objetos, nombres)
    if not tiles_mld:
        print("  Sin tiles MLD")
        errores += 1
        continue

    centros = np.array([
        ((t['bbox'][0]+t['bbox'][2])/2, (t['bbox'][1]+t['bbox'][3])/2)
        for t in tiles_mld
    ])

    # Buscar en el MLD el tile que coincide con las coordenadas del CSV
    # (garantiza que es el mismo tile que se eligio como representativo)
    indices_rep = set()
    for _, row in grupo.iterrows():
        cx_csv = (row['env_x1'] + row['env_x2']) / 2
        cy_csv = (row['env_y1'] + row['env_y2']) / 2
        dists  = np.sqrt((centros[:,0]-cx_csv)**2 + (centros[:,1]-cy_csv)**2)
        imin   = dists.argmin()
        if dists[imin] <= ts * TOLERANCIA_FRACCION:
            indices_rep.add(imin)
        else:
            print(f"  [sin match] {row.get('ID_Cuadrado_PCA','')} "
                  f"dist={dists[imin]:.4f}mm (umbral={ts*TOLERANCIA_FRACCION:.4f}mm)")

    print(f"  Tiles marcados: {len(indices_rep)}/{len(grupo)}")

    # Dibujo
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_aspect('equal')
    cats_usadas = set()

    for ann in sorted(anots, key=lambda o: o['w']*o['h'], reverse=True):
        cat = ann.get('categoria', 'otros')
        c, a, _ = COLOR_CAT.get(cat, COLOR_CAT['otros'])
        if c is None: continue
        cats_usadas.add(cat)
        if len(ann['points']) >= 3:
            ax.add_patch(MplPolygon(ann['points'], closed=True,
                                    facecolor=c, edgecolor=c, alpha=a, lw=0.3, zorder=2))

    for t in tiles_mld:
        if len(t['points']) >= 3:
            ax.add_patch(MplPolygon(t['points'], closed=True,
                                    lw=0.25, edgecolor='#888888', facecolor='none',
                                    alpha=0.5, zorder=3))

    for idx in indices_rep:
        if len(tiles_mld[idx]['points']) >= 3:
            ax.add_patch(MplPolygon(tiles_mld[idx]['points'], closed=True,
                                    lw=2.0, edgecolor='#E67E00', facecolor='#FF8C00',
                                    alpha=0.8, zorder=5))

    xs1 = [t['bbox'][0] for t in tiles_mld]; xs2 = [t['bbox'][2] for t in tiles_mld]
    ys1 = [t['bbox'][1] for t in tiles_mld]; ys2 = [t['bbox'][3] for t in tiles_mld]
    m = ts * 3
    ax.set_xlim(min(xs1)-m, max(xs2)+m)
    ax.set_ylim(min(ys1)-m, max(ys2)+m)

    ley = []
    for cat in ['tumor', 'stroma', 'muscle', 'otros']:
        if cat not in cats_usadas: continue
        c, a, nom = COLOR_CAT[cat]
        if c: ley.append(mpatches.Patch(facecolor=c, alpha=a, label=nom))
    ley += [
        mpatches.Patch(facecolor='none', edgecolor='#888888', lw=0.8,
                       label=f'Todos los tiles ({len(tiles_mld)})'),
        mpatches.Patch(facecolor='#FF8C00', alpha=0.8,
                       label=f'Representativos PCA ({len(indices_rep)})'),
    ]
    ax.legend(handles=ley, loc='upper right', fontsize=9)
    ax.set_title(f"Solapamiento MLD — {case}", fontsize=13, fontweight='bold')
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    plt.tight_layout()

    out_png = out(f"solapamiento_mld/{case}_solapamiento.png")
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()
    procesados += 1
    print(f"  Guardado: {out_png}")

print(f"""
{'='*50}
SOLAPAMIENTO COMPLETADO
Procesados ahora : {procesados}
Ya existían      : {len(ya_hechos)}
Errores/sin MLD  : {errores}
Carpeta          : {out("solapamiento_mld")}
{'='*50}
""")


#py -3.12 c:/Users/carme/OneDrive/Escritorio/M.UCM/TFM/1700MICRAS/CENTROIDE/A2_EDA.py

