

#  CONFIGURACIÓN

DF_ALL_CSV    = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\1700MICRAS\CENTROIDE\FEATURE_ENGINEERING\new_eda2\df_all_completo.csv"
GUIA_CSV      = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\1700MICRAS\CENTROIDE\FEATURE_ENGINEERING\new_eda2\GUIA_CENTROIDE.csv"
MLD_FOLDER    = r"\\imgserver\IMAGES\CONFOCAL\IA\crodriguezj\annotations\1700_microns"
CZI_FOLDER    = r"\\imgserver\IMAGES\CONFOCAL\IA\crodriguezj\images"
OUTPUT_FOLDER = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\1700MICRAS\CENTROIDE\FEATURE_ENGINEERING\TUCKER\FINAL"

# CASOS_SOLAPAMIENTO = [
#     'B13-03049',
#     'B16-03778',
#     'B12-09742',
#     'B01-03393',
#     'B16-11640',
# ]

CASOS_SOLAPAMIENTO = [
    # 'B-250005057_F31',
    # 'B-250009903_A37',
    # 'B-250009904_C23',
    # 'B-250009977_C41',
    # 'B00-07097',
    # 'B0000551',
    # 'B01-01611',
    # 'B01-03393',
    # 'B01-06901',
    # 'B09-00521',
    # 'B09-06164',
    # 'B09-08851',
    # 'B10-01993',
    # 'B12-00810',
    # 'B12-03804',
    # 'B12-07624',
    # 'B12-08038',
    # 'B12-09439',
    # 'B12-09742',
    # 'B12-13581',
    # 'B12-14477',
    # 'B13-02873',
    # 'B13-03049',
    # 'B13-10751',
    # 'B13-11488',
    # 'B14-00706',
    # 'B14-01463',
    # 'B14-05631',
    # 'B14-06707',
    # 'B14-07500',
    # 'B15-02633',
    # 'B15-10417',
    # 'B15-14792',
    # 'B16-00569',
    # 'B16-03778',
    # 'B16-09573',
    # 'B16-11640',
    # 'B16-14753',
    # 'B17-02079',
    # 'B17-02200',
    # 'B17-04590',
    # 'B17-05483',
    # 'B17-08213',
    # 'B17-10365',
    # 'B17-11302',
    # 'B17-11640',
    # 'B17-12348',
    # 'B17-13914',
    # 'B18-10263',
    # 'B18-10441',
    # 'B18-13221',
    # 'B18-14944',
    # 'B18-2088',
    # 'B18-3133',
    # 'B18-5575',
    # 'B18-7166',
    # 'B18-9285',
    # 'B20-10696',
    # 'B20-1733',
    # 'B20-3827',
    # 'B20-5474',
    # 'B21-11569',
    # 'B21-4930',
    # 'B21-7571',
    # 'B22-2857',
    # 'B22-4428',
    # 'B22-5640',
    # 'B23-10574',
    # 'B23-14939',
    # 'B23-1599',
    # 'B23-3723',
    # 'B23-3950',
    # 'B23-7097',
    # 'B25-00019',
    # 'M02-08934',
    # 'M02-09368',
    'M07-02604',
]


N_BINS       = 10
N_PAT        = 5
PIXEL_SIZE_UM = 0.1723
ZOOM_CZI     = 0.15
CELDA_PX     = 160     # tamaño celda del grid

# Mosaico
MAX_CROPS_MOSAICO = 50 # crops por patrón como máximo
CROP_SIZE_PX = (200, 200)   # tamaño de cada miniatura en el mosaico

# Grosor del recuadro en el solapamiento (en mm)
GROSOR_RECUADRO_MM = 0.03

COLORS_PAT = ['#2ECC71', '#F39C12', '#E74C3C', '#9B59B6', '#3498DB']


COLOR_CAT = {
    "tumor":  ("#E74833", 0.35, "Tumor"),        # VERDE
    "stroma": ("#34DB50", 0.30, "No músculo/No tumor"),  # azul
    "muscle": ("#361EE9", 0.35, "Músculo"),       # ROSA
    "bk":     (None,      0.0,  "Background"),
    "otros":  ('#FF00FF', 0.25, "Otros"),
}
LABEL_NUM = {1:"stroma", 2:"tumor", 3:"bk", 4:"stroma",
             5:"muscle", 6:"stroma", 7:"stroma", 8:"stroma", 9:"stroma"}
TID_DEF   = {0:"stroma", 1:"stroma", 2:"tumor", 3:"bk", 4:"muscle", 5:"muscle"}

# IMPORTS

import os, re, struct, warnings, xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from PIL import ImageDraw
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon, Rectangle as MplRect
from matplotlib.colors import to_rgba
from PIL import Image

import tensorly as tl
from tensorly.decomposition import non_negative_tucker

warnings.filterwarnings('ignore')
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
def out(f): return os.path.join(OUTPUT_FOLDER, f)

try:
    from pylibCZIrw import czi as pyczi
    CZI_OK = True
except ImportError:
    pyczi = None; CZI_OK = False
    print("[AVISO] pylibCZIrw no disponible ")

#  1. TUCKER

print("Cargando datos y calculando Tucker...")
df   = pd.read_csv(DF_ALL_CSV)
guia = pd.read_csv(GUIA_CSV, sep=';')
CASOS_CON_GUIA = set(guia['Case'].unique())

pc1_edges = np.linspace(df['PC1'].min(), df['PC1'].max(), N_BINS+1)
pc2_edges = np.linspace(df['PC2'].min(), df['PC2'].max(), N_BINS+1)
df['_b1'] = pd.cut(df['PC1'], bins=pc1_edges, labels=False, include_lowest=True).astype(int)
df['_b2'] = pd.cut(df['PC2'], bins=pc2_edges, labels=False, include_lowest=True).astype(int)

casos    = sorted(df['Case'].unique())
N_CASOS  = len(casos)
caso_idx = {c:i for i,c in enumerate(casos)}

tensor = np.zeros((N_CASOS, N_BINS, N_BINS))
for caso, grupo in df.groupby('Case'):
    i = caso_idx[caso]; total = len(grupo)
    for (b1,b2), cnt in grupo.groupby(['_b1','_b2']).size().items():
        tensor[i, int(b1), int(b2)] = cnt / total

tl.set_backend('numpy')
best_err, best_core, best_facs = np.inf, None, None
for seed in range(15):
    core, facs = non_negative_tucker(tensor, rank=[N_PAT,5,5],
                                      n_iter_max=500, random_state=seed, verbose=False)
    r   = tl.tucker_to_tensor((core, facs))
    err = np.linalg.norm(tensor-r)/np.linalg.norm(tensor)
    if err < best_err:
        best_err, best_core, best_facs = err, core, facs
print(f"  Tucker error: {best_err:.5f}")

F_patient = best_facs[0]; F_pc1 = best_facs[1]; F_pc2 = best_facs[2]
pattern_maps = []
for p in range(N_PAT):
    m = np.zeros((N_BINS, N_BINS))
    for k1 in range(5):
        for k2 in range(5):
            m += best_core[p,k1,k2] * np.outer(F_pc1[:,k1], F_pc2[:,k2])
    pattern_maps.append(m / (m.max()+1e-12))

F_norm   = F_patient / (F_patient.max(axis=0, keepdims=True)+1e-12)
pat_cols = [f'Patron_{p+1}' for p in range(N_PAT)]
df_c     = pd.DataFrame(F_norm, columns=pat_cols)
df_c['Case'] = casos

bin_patron_dom = {}
for b1 in range(N_BINS):
    for b2 in range(N_BINS):
        vals = [pattern_maps[p][b1,b2] for p in range(N_PAT)]
        bin_patron_dom[(b1,b2)] = int(np.argmax(vals))

df['_pat_dom'] = df.apply(lambda r: bin_patron_dom[(
    min(int(r['_b1']),N_BINS-1), min(int(r['_b2']),N_BINS-1))], axis=1)
df['_pat_intens'] = df.apply(lambda r: float(pattern_maps[r['_pat_dom']][
    min(int(r['_b1']),N_BINS-1), min(int(r['_b2']),N_BINS-1)]), axis=1)


#  2. FUNCIONES CZI (recorte de tiles)

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
        # Escena única: formula con CenterPosition
        cx_um, cy_um = get_center_um(czidoc)
        if cx_um is None: return None
        cx_px = (bbox['X'][0] + bbox['X'][1]) / 2
        cy_px = (bbox['Y'][0] + bbox['Y'][1]) / 2

        px_l = int(round((x1*1000 - cx_um) / pxum + cx_px))
        px_r = int(round((x2*1000 - cx_um) / pxum + cx_px))
        py_t = int(round(-(max(y1,y2)*1000 - cy_um) / pxum + cy_px))
        py_b = int(round(-(min(y1,y2)*1000 - cy_um) / pxum + cy_px))
        xs = min(px_l, px_r); ys = min(py_t, py_b)
        w  = abs(px_r - px_l); h  = abs(py_b - py_t)

        if not dentro(xs, ys, w, h):
            # Fallback: usar centro del bbox en píxeles directamente
            cx2 = (bbox['X'][0] + bbox['X'][1]) / 2 * pxum
            cy2 = (bbox['Y'][0] + bbox['Y'][1]) / 2 * pxum
            cx_px2 = (bbox['X'][0] + bbox['X'][1]) / 2
            cy_px2 = (bbox['Y'][0] + bbox['Y'][1]) / 2
            px_l = int(round((x1*1000 - cx2) / pxum + cx_px2))
            px_r = int(round((x2*1000 - cx2) / pxum + cx_px2))
            py_t = int(round(-(max(y1,y2)*1000 - cy2) / pxum + cy_px2))
            py_b = int(round(-(min(y1,y2)*1000 - cy2) / pxum + cy_px2))
            xs = min(px_l, px_r); ys = min(py_t, py_b)
            w  = abs(px_r - px_l); h  = abs(py_b - py_t)
            if not dentro(xs, ys, w, h): return None

    else:
        # Multi-escena 3 intentos en orden

        # Intento 1: Y invertido respecto al borde inferior del bbox total
        total_y_bottom = bbox['Y'][1]
        xs = min(int(round(x1*1000/pxum)), int(round(x2*1000/pxum)))
        ys = min(int(round(-(y1*1000/pxum) + total_y_bottom)),
                 int(round(-(y2*1000/pxum) + total_y_bottom)))
        w  = abs(int(round(x2*1000/pxum)) - int(round(x1*1000/pxum)))
        h  = abs(int(round(-(y2*1000/pxum) + total_y_bottom)) -
                 int(round(-(y1*1000/pxum) + total_y_bottom)))

        if not dentro(xs, ys, w, h):
            # Intento 2: CenterPosition
            cx_um, cy_um = get_center_um(czidoc)
            if cx_um is not None:
                cx_px = (bbox['X'][0] + bbox['X'][1]) / 2
                cy_px = (bbox['Y'][0] + bbox['Y'][1]) / 2
                px_l = int(round((x1*1000 - cx_um) / pxum + cx_px))
                px_r = int(round((x2*1000 - cx_um) / pxum + cx_px))
                py_t = int(round(-(max(y1,y2)*1000 - cy_um) / pxum + cy_px))
                py_b = int(round(-(min(y1,y2)*1000 - cy_um) / pxum + cy_px))
                xs = min(px_l, px_r); ys = min(py_t, py_b)
                w  = abs(px_r - px_l); h  = abs(py_b - py_t)

            if not dentro(xs, ys, w, h):
                # Intento 3: conversión directa sin invertir Y
                xs = min(int(round(x1*1000/pxum)), int(round(x2*1000/pxum)))
                ys = min(int(round(y1*1000/pxum)), int(round(y2*1000/pxum)))
                w  = abs(int(round(x2*1000/pxum)) - int(round(x1*1000/pxum)))
                h  = abs(int(round(y2*1000/pxum)) - int(round(y1*1000/pxum)))
                if not dentro(xs, ys, w, h): return None

    reg = czidoc.read(roi=(xs, ys, w, h), zoom=zoom)
    if reg is None or reg.size == 0 or reg.max() == 0: return None  
    # Filtro adicional: descartar si más del 20% de píxeles es fondo vacío

    img_arr = reg[..., :3]  # solo RGB
    n_pixeles = img_arr.shape[0] * img_arr.shape[1]
    n_negro  = np.sum(img_arr.max(axis=2) < 15)
    n_blanco = np.sum(img_arr.min(axis=2) > 240)
    if (n_negro + n_blanco) / n_pixeles > 0.10:
        return None
    img = Image.fromarray(reg[..., ::-1].astype(np.uint8))
    return img.resize((int(round(w*zoom)), int(round(h*zoom))), Image.LANCZOS)
    
#  3. MOSAICO DE CROPS POR PATRÓN


col_x1 = 'Object Info (tile) - Envelope left'
col_y1 = 'Object Info (tile) - Envelope top'
col_x2 = 'Object Info (tile) - Envelope right'
col_y2 = 'Object Info (tile) - Envelope bottom'

for p in range(N_PAT):
    col = COLORS_PAT[p]
    print(f"\nPatrón {p+1}...")

    # Ordenar casos por score
    ranking = (df_c.sort_values(f'Patron_{p+1}', ascending=False)
                   .loc[df_c['Case'].isin(CASOS_CON_GUIA)])

    crops_recopilados = []   # lista de PIL Images con metadata

    for _, row_caso in ranking.iterrows():
        if len(crops_recopilados) >= MAX_CROPS_MOSAICO:
            break
        caso  = row_caso['Case']
        score = row_caso[f'Patron_{p+1}']

        # Todos los tiles de este caso que caen en bins donde P domina
        df_caso = df[df['Case'] == caso].copy()
        df_caso_p = df_caso[df_caso['_pat_dom'] == p]

        if df_caso_p.empty:
            continue

        if not CZI_OK:
            break

        czi_path = Path(CZI_FOLDER) / f"{caso}.czi"
        if not czi_path.exists():
            continue

        try:
            with pyczi.open_czi(str(czi_path)) as czidoc:
            
                # Ordenar por intensidad descendente (los más representativos primero)
                df_caso_p_sorted = df_caso_p.sort_values('_pat_intens', ascending=False)
                for _, tile_row in df_caso_p_sorted.iterrows():
                    if len(crops_recopilados) >= MAX_CROPS_MOSAICO:
                        break
                    try:
                        img = recortar_tile_czi(
                            czidoc,
                            float(tile_row[col_x1]), float(tile_row[col_y1]),
                            float(tile_row[col_x2]), float(tile_row[col_y2]),
                            pxum=PIXEL_SIZE_UM, zoom=ZOOM_CZI
                        )
                        if img is not None:
                            img_r = img.resize(CROP_SIZE_PX, Image.LANCZOS)
                            crops_recopilados.append({
                                'img': img_r, 'caso': caso, 'score': score,
                                'intens': float(tile_row['_pat_intens'])
                            })
                    except Exception:
                        pass
        except Exception as e:
            print(f"  [ERROR CZI {caso}] {e}")
            continue

    n_crops = len(crops_recopilados)
    print(f"  {n_crops} crops recopilados")

    if n_crops == 0:
        print(f"  Sin crops en {p+1}")
        continue

    #  Montar mosaico 
    ncols_m = min(8, n_crops)
    nrows_m = (n_crops + ncols_m - 1) // ncols_m

    W = CROP_SIZE_PX[0]; H = CROP_SIZE_PX[1]
    gap = 4   # píxeles entre crops

    # Color de borde del patrón
    r_c = int(col[1:3],16); g_c = int(col[3:5],16); b_c = int(col[5:7],16)

    mosaic_w = ncols_m * W + (ncols_m-1) * gap
    mosaic_h = nrows_m * H + (nrows_m-1) * gap
    mosaic = Image.new('RGB', (mosaic_w, mosaic_h), (30,30,30))

    for k, d in enumerate(crops_recopilados):
        row_m = k // ncols_m; col_m = k % ncols_m
        x_off = col_m * (W + gap); y_off = row_m * (H + gap)
        mosaic.paste(d['img'], (x_off, y_off))

        # Borde de color del patrón
        from PIL import ImageDraw as PILDraw
        draw_m = PILDraw.Draw(mosaic)
        bw = 3  # grosor del borde
        draw_m.rectangle(
            [x_off, y_off, x_off+W-1, y_off+H-1],
            outline=(r_c, g_c, b_c), width=bw
        )
        #
        label = f"{d['caso'][-8:]} {d['score']:.2f}"
        draw_m.rectangle([x_off, y_off+H-14, x_off+W, y_off+H],
                          fill=(0,0,0))
        draw_m.text((x_off+2, y_off+H-13), label, fill=(255,255,255))

    # Guardar con matplotlib para añadir título
    fig, ax = plt.subplots(figsize=(ncols_m*2.2, nrows_m*2.2+0.8),
                            facecolor='white')
    ax.imshow(np.array(mosaic))
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(col); spine.set_linewidth(3)
    plt.title(f'Patrón {p+1} — Mosaico de tiles ({n_crops} crops)\n'
              f'Ordenados por score Tucker descendente',
              fontsize=13, fontweight='bold', color=col, pad=8)
    plt.tight_layout()
    plt.savefig(out(f"MOSAICO_patron{p+1}.png"),
                dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  MOSAICO_patron{p+1}.png ({nrows_m}×{ncols_m} grid)")

#  4. FUNCIONES MLD
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
            lname = ln.get("Name",""); res[lname] = {}
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
                        shape=buf[off]; off+=1; ttype=buf[off]; off+=1
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
    roi_cand=[o for o in objetos if o['layer']=='ROI' and 0.2<o['w']<2.5 and abs(o['w']-o['h'])<0.2]
    tid=Counter(o['type_id'] for o in roi_cand).most_common(1)[0][0] if roi_cand else None
    tiles=[o for o in objetos if o['layer']=='ROI' and o['type_id']==tid and o['w']<3.0]
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


#5. SOLAPAMIENTO CON RECUADROS
ZOOM_FONDO = 0.15




def construir_fondo_czi(czidoc, df_caso, canvas_mm):
    """
    Construye un canvas PIL pegando cada tile en su posición mm real.
    Devuelve (img_canvas, px_por_mm).
    """
    tile_mm   = float(df_caso[col_x2].iloc[0] - df_caso[col_x1].iloc[0])
    tile_px   = int(round(tile_mm * 1000 / PIXEL_SIZE_UM * ZOOM_FONDO))
    px_por_mm = tile_px / tile_mm

    w_mm = canvas_mm['x_max'] - canvas_mm['x_min']
    h_mm = canvas_mm['y_max'] - canvas_mm['y_min']
    W_px = max(1, int(round(w_mm * px_por_mm)))
    H_px = max(1, int(round(h_mm * px_por_mm)))

    canvas = Image.new('RGB', (W_px, H_px), (240, 240, 240))

    n_ok = 0
    for _, tile_row in df_caso.iterrows():
        tx1 = float(tile_row[col_x1]); ty1 = float(tile_row[col_y1])
        tx2 = float(tile_row[col_x2]); ty2 = float(tile_row[col_y2])

        try:
            img_tile = recortar_tile_czi(
                czidoc, tx1, ty1, tx2, ty2,
                pxum=PIXEL_SIZE_UM, zoom=ZOOM_FONDO)
        except Exception:
            continue
        if img_tile is None:
            continue


        y_top = min(ty1, ty2)   # el borde con menor Y es el de arriba
        px_x  = int(round((tx1 - canvas_mm['x_min']) * px_por_mm))
        px_y  = int(round((y_top - canvas_mm['y_min']) * px_por_mm))

        tile_w_px = max(1, int(round(abs(tx2 - tx1) * px_por_mm)))
        tile_h_px = max(1, int(round(abs(ty2 - ty1) * px_por_mm)))
        img_tile_r = img_tile.resize((tile_w_px, tile_h_px), Image.LANCZOS)

        if px_x < W_px and px_y < H_px:
            canvas.paste(img_tile_r, (max(0, px_x), max(0, px_y)))
            n_ok += 1

    print(f"  Tiles pegados en canvas: {n_ok}/{len(df_caso)}")
    return canvas, px_por_mm


def añadir_recuadros_tucker(ax, df_caso, con_czi=False):
    """Dibuja los recuadros Tucker sobre el eje ax."""
    for _, row in df_caso.iterrows():
        x1 = float(row[col_x1]); y1 = float(row[col_y1])
        x2 = float(row[col_x2]); y2 = float(row[col_y2])
        p_dom  = int(row['_pat_dom'])
        intens = float(row['_pat_intens'])

        col_hex = COLORS_PAT[p_dom]
        r_c, g_c, b_c, _ = to_rgba(col_hex)

        log_min = np.log10(0.001); log_max = 0.0
        if intens >= 0.001:
            t_val = float(np.clip(
                (np.log10(intens) - log_min) / (log_max - log_min), 0, 1))
            lw_tile     = (1.5 + t_val * 3.5) if con_czi else (0.4 + t_val * 2.5)
            alpha_borde = (0.9 + t_val * 0.09) if con_czi else (0.4 + t_val * 0.55)
        else:
            lw_tile = 0.4; alpha_borde = 0.35

        pts = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
        ax.add_patch(MplPolygon(pts, closed=True,
                                facecolor='none',
                                edgecolor=(r_c, g_c, b_c, alpha_borde),
                                lw=lw_tile, zorder=4))


def leyenda_tucker(df_caso):
    ley = []
    for p in range(N_PAT):
        cnt = (df_caso['_pat_dom'] == p).sum()
        if cnt == 0: continue
        pct = cnt / len(df_caso) * 100
        ley.append(mpatches.Patch(facecolor='none',
                                   edgecolor=COLORS_PAT[p], lw=2.0,
                                   label=f'Patrón {p+1} ({cnt} tiles, {pct:.0f}%)'))
    return ley



for caso in CASOS_SOLAPAMIENTO:
    print(f"\nCaso: {caso}")

    df_caso = df[df['Case'] == caso].copy()
    if df_caso.empty:
        print(f"  [AVISO] Caso no encontrado"); continue
    print(f"  Tiles: {len(df_caso)}")

    mld_path = Path(MLD_FOLDER) / f"{caso}.mld"
    if not mld_path.exists():
        print(f"  [AVISO] MLD no encontrado"); continue

    nombres = leer_layer_configs(str(mld_path))
    try:
        objetos  = leer_mld(str(mld_path))
        tiles_mld, anots, ts = separar_mld(objetos, nombres)
    except Exception as e:
        print(f"  [ERROR MLD] {e}"); continue

    F_vals = best_facs[0][caso_idx.get(caso, 0)] if caso in caso_idx else None
    scores_str = ('  '.join([f'P{i+1}={v:.2f}' for i, v in enumerate(F_vals)])
                  if F_vals is not None else '')

    mg = float(df_caso[col_x2].iloc[0] - df_caso[col_x1].iloc[0]) * 2
    xlim = (df_caso[col_x1].min()-mg, df_caso[col_x2].max()+mg)
    ylim = (df_caso[col_y2].max()+mg, df_caso[col_y1].min()-mg)

    # IMAGEN A: MLD + recuadros Tucker 
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
                                    facecolor=c, edgecolor=c,
                                    alpha=a, lw=0.3, zorder=1))

    for t in tiles_mld:
        if len(t['points']) >= 3:
            ax.add_patch(MplPolygon(t['points'], closed=True,
                                    lw=0.2, edgecolor='#BBBBBB',
                                    facecolor='none', alpha=0.5, zorder=2))

    añadir_recuadros_tucker(ax, df_caso, con_czi=False)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel('X (mm)', fontsize=11); ax.set_ylabel('Y (mm)', fontsize=11)
    ax.grid(True, alpha=0.08, lw=0.4)

    ley_a = []
    for cat in ['muscle','tumor','stroma']:
        if cat not in cats_usadas: continue
        c, a, nom = COLOR_CAT[cat]
        if c: ley_a.append(mpatches.Patch(facecolor=c, alpha=max(a,0.5), label=nom))
    ley_a += leyenda_tucker(df_caso)
    ax.legend(handles=ley_a, loc='upper right', fontsize=9,
              framealpha=0.92, title='Leyenda', title_fontsize=10)
    ax.set_title(f'Solapamiento Tucker — {caso}\nScores: {scores_str}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fname_a = out(f"SOLAPAMIENTO_recuadros_{caso}.png")
    plt.savefig(fname_a, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   {Path(fname_a).name}  (MLD)")

    # IMAGEN B: CZI entero como fondo 
    if not CZI_OK:
        print(f"  [INFO] pylibCZIrw no disponible"); continue

    czi_path = Path(CZI_FOLDER) / f"{caso}.czi"
    if not czi_path.exists():
        print(f"  [INFO] CZI no encontrado"); continue

    try:
        print(f"  Leyendo CZI completo...")
        with pyczi.open_czi(str(czi_path)) as czidoc:
            bbox = czidoc.total_bounding_box
            W_czi = bbox['X'][1] - bbox['X'][0]
            H_czi = bbox['Y'][1] - bbox['Y'][0]
            reg = czidoc.read(roi=(bbox['X'][0], bbox['Y'][0], W_czi, H_czi),
                              zoom=ZOOM_FONDO)
            img_czi = Image.fromarray(reg[..., ::-1].astype(np.uint8))

            cx_um, cy_um = get_center_um(czidoc)
            cx_px = (bbox['X'][0] + bbox['X'][1]) / 2
            cy_px = (bbox['Y'][0] + bbox['Y'][1]) / 2

            x_czi_min_mm = (bbox['X'][0] - cx_px) * PIXEL_SIZE_UM / 1000 + cx_um / 1000
            x_czi_max_mm = (bbox['X'][1] - cx_px) * PIXEL_SIZE_UM / 1000 + cx_um / 1000
            y_czi_min_mm = -(bbox['Y'][1] - cy_px) * PIXEL_SIZE_UM / 1000 + cy_um / 1000
            y_czi_max_mm = -(bbox['Y'][0] - cy_px) * PIXEL_SIZE_UM / 1000 + cy_um / 1000

        print(f"  CZI OK: {img_czi.size[0]}×{img_czi.size[1]}px")
        print(f"  Extent mm: X[{x_czi_min_mm:.2f}, {x_czi_max_mm:.2f}]  Y[{y_czi_min_mm:.2f}, {y_czi_max_mm:.2f}]")

    except Exception as e:
        print(f"  [AVISO] Error leyendo CZI: {e}"); continue

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_aspect('equal')

    ax.imshow(np.array(img_czi),
              extent=[x_czi_min_mm, x_czi_max_mm, y_czi_min_mm, y_czi_max_mm],
              origin='upper', aspect='auto', zorder=0)

    añadir_recuadros_tucker(ax, df_caso, con_czi=True)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel('X (mm)', fontsize=11); ax.set_ylabel('Y (mm)', fontsize=11)
    ax.grid(True, alpha=0.08, lw=0.4)

    ley_b = [mpatches.Patch(facecolor='#DDDDDD', edgecolor='gray',
                             label='Fondo: imagen CZI real')]
    ley_b += leyenda_tucker(df_caso)
    ax.legend(handles=ley_b, loc='upper right', fontsize=9,
              framealpha=0.92, title='Leyenda', title_fontsize=10)
    ax.set_title(f'Tiles Tucker sobre imagen CZI — {caso}\nScores: {scores_str}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fname_b = out(f"SOLAPAMIENTO_recuadros_{caso}_CZI.png")
    plt.savefig(fname_b, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   {Path(fname_b).name}  (CZI)")


#6. GRID DE THUMBNAILS + OVERLAY TUCKER (del tucker_overlay)


from scipy.ndimage import gaussian_filter


GRID_PNG   = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\1700MICRAS\CENTROIDE\FEATURE_ENGINEERING\new_eda2\08_grid_thumbnails_PCA.png"
CELDA_PX   = 160
ALPHA_MAX  = 180
ALPHA_MIN  = 15
UMBRAL_PM  = 0.001

gs = N_BINS * CELDA_PX

if Path(GRID_PNG).exists():
    grid_base = Image.open(GRID_PNG).convert("RGBA")
    if grid_base.size != (gs, gs):
        grid_base = grid_base.resize((gs, gs), Image.LANCZOS)

    # Tucker para el grid
    F_norm_g = best_facs[0] / (best_facs[0].max(axis=0, keepdims=True)+1e-12)
    pat_cols_g = [f"Patron_{p+1}" for p in range(N_PAT)]
    df_c_g = pd.DataFrame(F_norm_g, columns=pat_cols_g)
    df_c_g["Case"] = casos

    mean_density = tensor.mean(axis=0)
    entropias_g = []
    for p in range(N_PAT):
        pf = pattern_maps[p].flatten(); pf=pf[pf>0]; pf/=pf.sum()
        entropias_g.append(-np.sum(pf*np.log2(pf+1e-12)))

    bins_con_tile_g = set(zip(guia["bin_col"].astype(int), guia["bin_row"].astype(int)))

    def construir_grid(p):
        r_c,g_c,b_c = [int(COLORS_PAT[p][i:i+2],16) for i in (1,3,5)]
        resultado = grid_base.copy()
        overlay = Image.new("RGBA",(gs,gs),(0,0,0,0))
        draw_ov = ImageDraw.Draw(overlay)
        log_min = np.log10(UMBRAL_PM); log_max = 0.0
        for b1 in range(N_BINS):
            for b2 in range(N_BINS):
                intens = float(pattern_maps[p][b1,b2])
                if intens < UMBRAL_PM: continue
                t = float(np.clip((np.log10(intens)-log_min)/(log_max-log_min),0,1))
                ab = int(ALPHA_MIN + t*(ALPHA_MAX-ALPHA_MIN))
                xoff=b1*CELDA_PX; yoff=(N_BINS-1-b2)*CELDA_PX
                draw_ov.rectangle([xoff,yoff,xoff+CELDA_PX-1,yoff+CELDA_PX-1],
                                   fill=(r_c,g_c,b_c,ab))
        resultado = Image.alpha_composite(resultado, overlay)
        draw_bd = ImageDraw.Draw(resultado)
        seg=12; gap=6; grosor=2
        for b1 in range(N_BINS):
            for b2 in range(N_BINS):
                if (b1,b2) in bins_con_tile_g: continue
                if mean_density[b1,b2] < mean_density.max()*0.05: continue
                xoff=b1*CELDA_PX; yoff=(N_BINS-1-b2)*CELDA_PX
                c=(150,150,150,180)
                for x in range(xoff,xoff+CELDA_PX,seg+gap):
                    xe=min(x+seg,xoff+CELDA_PX)
                    draw_bd.line([(x,yoff),(xe,yoff)],fill=c,width=grosor)
                    draw_bd.line([(x,yoff+CELDA_PX-1),(xe,yoff+CELDA_PX-1)],fill=c,width=grosor)
                for y in range(yoff,yoff+CELDA_PX,seg+gap):
                    ye=min(y+seg,yoff+CELDA_PX)
                    draw_bd.line([(xoff,y),(xoff,ye)],fill=c,width=grosor)
                    draw_bd.line([(xoff+CELDA_PX-1,y),(xoff+CELDA_PX-1,ye)],fill=c,width=grosor)
        return resultado.convert("RGB")

    def añadir_contorno(ax):
        dens_s = gaussian_filter(mean_density, sigma=0.9)
        if dens_s.max()==0: return
        dn = dens_s/dens_s.max()
        X = np.arange(N_BINS)*CELDA_PX+CELDA_PX/2
        Y = (N_BINS-1-np.arange(N_BINS))*CELDA_PX+CELDA_PX/2
        Xg,Yg = np.meshgrid(X,Y); Z=dn.T
        ax.contour(Xg,Yg,Z,levels=[0.08,0.2,0.4,0.65,0.85],
                   colors="#CC66CC",linewidths=1.0,alpha=0.7,zorder=5)

    # Caso representativo de cada patrón
    casos_repr = {}
    for p in range(N_PAT):
        rank = df_c_g.sort_values(f"Patron_{p+1}", ascending=False)
        for _,row in rank.iterrows():
            if row["Case"] in CASOS_CON_GUIA:
                casos_repr[p] = (row["Case"], row[f"Patron_{p+1}"])
                break

    # Scatter colores por bin
    df["_pat_color_g"] = df.apply(
        lambda r: COLORS_PAT[bin_patron_dom[(min(int(r["_b1"]),N_BINS-1),
                                              min(int(r["_b2"]),N_BINS-1))]], axis=1)

    import matplotlib.patches as mpatches_g
    import matplotlib.gridspec as gridspec_g

    # FIG A: 5 grids solos
    fig, axes = plt.subplots(1, N_PAT, figsize=(5*N_PAT, 6), facecolor="white")
    for p, ax in enumerate(axes):
        col = COLORS_PAT[p]
        img_g = construir_grid(p)
        ax.imshow(np.array(img_g), origin="upper", extent=[0,gs,gs,0])
        añadir_contorno(ax)
        for i in range(N_BINS+1):
            ax.axvline(i*CELDA_PX,color="white",lw=0.35,alpha=0.4,zorder=6)
            ax.axhline(i*CELDA_PX,color="white",lw=0.35,alpha=0.4,zorder=6)
        ax.set_xlim(0,gs); ax.set_ylim(gs,0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Patrón {p+1}\nH={entropias_g[p]:.2f} bits",
                     fontsize=12,fontweight="bold",color=col,pad=8)
        ax.set_xlabel("PC1 ",fontsize=9)
        if p==0: ax.set_ylabel(" PC2",fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(col); spine.set_linewidth(3)
    plt.suptitle(f"Patrones Morfológicos Tucker — Atlas de Vejiga\n"
                 f"Rango [5,5,5] | Error={best_err:.3f} | n={N_CASOS} casos",
                 fontsize=11,fontweight="bold",y=1.02)
    plt.tight_layout(w_pad=1.5)
    plt.savefig(out("FIG_GRID_todos_patrones.png"),dpi=200,bbox_inches="tight",facecolor="white")
    plt.close(); print("   FIG_GRID_todos_patrones.png")

    # FIG B: grid solo por patrón
    for p in range(N_PAT):
        col=COLORS_PAT[p]; img_g=construir_grid(p)
        fig,ax=plt.subplots(figsize=(8,8.5),facecolor="white")
        ax.imshow(np.array(img_g),origin="upper",extent=[0,gs,gs,0])
        añadir_contorno(ax)
        for i in range(N_BINS+1):
            ax.axvline(i*CELDA_PX,color="white",lw=0.4,alpha=0.4,zorder=6)
            ax.axhline(i*CELDA_PX,color="white",lw=0.4,alpha=0.4,zorder=6)
        ax.set_xlim(0,gs); ax.set_ylim(gs,0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("PC1 ",fontsize=11); ax.set_ylabel(" PC2",fontsize=11)
        for spine in ax.spines.values():
            spine.set_edgecolor(col); spine.set_linewidth(4)
        plt.title(f"Patrón {p+1}  |  H={entropias_g[p]:.2f} bits",
                  fontsize=14,fontweight="bold",color=col,pad=10)
        plt.tight_layout()
        plt.savefig(out(f"FIG_GRID_SOLO_patron{p+1}.png"),dpi=220,bbox_inches="tight",facecolor="white")
        plt.close(); print(f"   FIG_GRID_SOLO_patron{p+1}.png")

    # FIG C: grid + scatter por patrón
    for p in range(N_PAT):
        col=COLORS_PAT[p]
        if p not in casos_repr: continue
        caso,score=casos_repr[p]
        img_g=construir_grid(p)
        fig=plt.figure(figsize=(16,7.5),facecolor="white")
        gs_fig=gridspec_g.GridSpec(1,2,figure=fig,width_ratios=[1,1.4],wspace=0.08)
        ax_grid=fig.add_subplot(gs_fig[0])
        ax_grid.imshow(np.array(img_g),origin="upper",extent=[0,gs,gs,0])
        añadir_contorno(ax_grid)
        for i in range(N_BINS+1):
            ax_grid.axvline(i*CELDA_PX,color="white",lw=0.4,alpha=0.4,zorder=6)
            ax_grid.axhline(i*CELDA_PX,color="white",lw=0.4,alpha=0.4,zorder=6)
        ax_grid.set_xlim(0,gs); ax_grid.set_ylim(gs,0)
        ax_grid.set_xticks([]); ax_grid.set_yticks([])
        ax_grid.set_xlabel("PC1 ",fontsize=10); ax_grid.set_ylabel("PC2",fontsize=10)
        ax_grid.set_title(f"Patrón {p+1}  |  H={entropias_g[p]:.2f} bits\n"
                          f"Grid tiles representativos + overlay Tucker",
                          fontsize=11,fontweight="bold",color=col,pad=8)
        for spine in ax_grid.spines.values():
            spine.set_edgecolor(col); spine.set_linewidth(3)
        ax_sc=fig.add_subplot(gs_fig[1])
        otros=df[df["Case"]!=caso]
        ax_sc.scatter(otros["PC1"],otros["PC2"],c="#CCCCCC",s=1.5,alpha=0.10,
                      rasterized=True,zorder=1,linewidths=0)
        caso_df=df[df["Case"]==caso]
        ax_sc.scatter(caso_df["PC1"],caso_df["PC2"],c=caso_df["_pat_color_g"],
                      s=16,alpha=0.75,zorder=2,edgecolors="none",rasterized=True)
        ax_sc.set_xlabel("PC1",fontsize=10); ax_sc.set_ylabel("PC2",fontsize=10)
        ax_sc.grid(True,alpha=0.2,lw=0.5); ax_sc.set_facecolor("#FAFAFA")
        handles_sc=[mpatches_g.Patch(facecolor=COLORS_PAT[pp],label=f"P{pp+1}",
                                      edgecolor="white",lw=0.5) for pp in range(N_PAT)]
        ax_sc.legend(handles=handles_sc,fontsize=7,loc="upper right",framealpha=0.9,
                     title="Patrón bin",title_fontsize=7)
        ax_sc.text(0.02,0.02,f"{caso}\nn={len(caso_df):,} tiles\nscore={score:.2f}",
                   transform=ax_sc.transAxes,ha="left",va="bottom",fontsize=7.5,
                   color=col,fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.35",fc="white",alpha=0.88,edgecolor=col,lw=1))
        for spine in ax_sc.spines.values():
            spine.set_edgecolor(col); spine.set_linewidth(1.5)
        ax_sc.set_title(f"Tiles de {caso} en espacio PCA\n(coloreados por patrón dominante del bin)",
                        fontsize=10,fontweight="bold")
        plt.suptitle(f"Patrón {p+1} — {caso}",fontsize=13,fontweight="bold",color=col,y=1.01)
        plt.tight_layout()
        plt.savefig(out(f"FIG_COMPLETA_patron{p+1}_{caso}.png"),dpi=200,bbox_inches="tight",facecolor="white")
        plt.close(); print(f"   FIG_COMPLETA_patron{p+1}_{caso}.png")

    # FIG D: mapa dominante
    resultado_dom=grid_base.copy()
    overlay_dom=Image.new("RGBA",(gs,gs),(0,0,0,0))
    draw_dom=ImageDraw.Draw(overlay_dom)
    for b1 in range(N_BINS):
        for b2 in range(N_BINS):
            pd_=bin_patron_dom[(b1,b2)]
            intens=float(pattern_maps[pd_][b1,b2])
            if intens<0.03: continue
            r_c,g_c,b_c=[int(COLORS_PAT[pd_][i:i+2],16) for i in (1,3,5)]
            xoff=b1*CELDA_PX; yoff=(N_BINS-1-b2)*CELDA_PX
            draw_dom.rectangle([xoff,yoff,xoff+CELDA_PX-1,yoff+CELDA_PX-1],
                                fill=(r_c,g_c,b_c,int(110*intens)))
    resultado_dom=Image.alpha_composite(resultado_dom,overlay_dom)
    draw_bd2=ImageDraw.Draw(resultado_dom)
    for b1 in range(N_BINS):
        for b2 in range(N_BINS):
            if (b1,b2) in bins_con_tile_g: continue
            if mean_density[b1,b2]<mean_density.max()*0.05: continue
            xoff=b1*CELDA_PX; yoff=(N_BINS-1-b2)*CELDA_PX
            c=(150,150,150,180)
            for x in range(xoff,xoff+CELDA_PX,12+6):
                xe=min(x+12,xoff+CELDA_PX)
                draw_bd2.line([(x,yoff),(xe,yoff)],fill=c,width=2)
                draw_bd2.line([(x,yoff+CELDA_PX-1),(xe,yoff+CELDA_PX-1)],fill=c,width=2)
            for y in range(yoff,yoff+CELDA_PX,12+6):
                ye=min(y+12,yoff+CELDA_PX)
                draw_bd2.line([(xoff,y),(xoff,ye)],fill=c,width=2)
                draw_bd2.line([(xoff+CELDA_PX-1,y),(xoff+CELDA_PX-1,ye)],fill=c,width=2)
    resultado_dom=resultado_dom.convert("RGB")
    fig,ax=plt.subplots(figsize=(10,10.5),facecolor="white")
    ax.imshow(np.array(resultado_dom),origin="upper",extent=[0,gs,gs,0])
    añadir_contorno(ax)
    for i in range(N_BINS+1):
        ax.axvline(i*CELDA_PX,color="white",lw=0.4,alpha=0.4,zorder=6)
        ax.axhline(i*CELDA_PX,color="white",lw=0.4,alpha=0.4,zorder=6)
    ax.set_xlim(0,gs); ax.set_ylim(gs,0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("PC1 ",fontsize=12); ax.set_ylabel(" PC2",fontsize=12)
    handles_dom=[mpatches_g.Patch(facecolor=COLORS_PAT[p],
                                   label=f"Patrón {p+1} (H={entropias_g[p]:.2f} bits)",
                                   edgecolor="white",lw=0.5) for p in range(N_PAT)]
    handles_dom.append(mpatches_g.Patch(facecolor="none",edgecolor="gray",
                                         linestyle="--",label="Bin sin representativo"))
    ax.legend(handles=handles_dom,fontsize=9,loc="upper right",framealpha=0.9,
              title="Patrón dominante",title_fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333"); spine.set_linewidth(2)
    plt.title(f"Mapa de Patrones Tucker — Patrón dominante por bin\n"
              f"Atlas de Vejiga | n={N_CASOS} casos",fontsize=13,fontweight="bold",pad=12)
    plt.tight_layout()
    plt.savefig(out("FIG_GRID_dominante.png"),dpi=200,bbox_inches="tight",facecolor="white")
    plt.close(); print("   FIG_GRID_dominante.png")

else:
    print(f"  [AVISO] Grid PNG no encontrado: {GRID_PNG}")

#  7. HEATMAP CONTRIBUCIONES + PIE + CORRELACIONES

import seaborn as sns

# df_c para estas figuras
pat_cols_a = [f"Patron_{p+1}" for p in range(N_PAT)]
F_norm_a = best_facs[0] / (best_facs[0].max(axis=0,keepdims=True)+1e-12)
df_ca = pd.DataFrame(F_norm_a, columns=pat_cols_a)
df_ca["Case"] = casos
df_ca["patron_dominante"] = df_ca[pat_cols_a].idxmax(axis=1)

# Heatmap
data_heat = df_ca[pat_cols_a].T
data_heat.columns = df_ca["Case"]
dom_idx = df_ca[pat_cols_a].idxmax(axis=1)
order_h = df_ca.assign(_dom=dom_idx).sort_values(
    [f"Patron_{p+1}" for p in range(N_PAT)], ascending=False).index
data_heat = data_heat[df_ca.loc[order_h,"Case"]]
fig,ax=plt.subplots(figsize=(min(N_CASOS*0.35+2,30),5))
sns.heatmap(data_heat,ax=ax,cmap="YlOrRd",vmin=0,vmax=1,xticklabels=True,
            yticklabels=[f"P{p+1}" for p in range(N_PAT)],
            cbar_kws={"label":"Contribución norm.","shrink":0.6},
            linewidths=0.3,linecolor="#CCCCCC")
ax.set_xticklabels(ax.get_xticklabels(),rotation=75,ha="right",fontsize=5.5)
for label,col in zip(ax.get_yticklabels(),COLORS_PAT):
    label.set_color(col)
ax.set_title(f"Contribución de Patrones Tucker por Caso (n={N_CASOS})",fontsize=13,fontweight="bold")
plt.tight_layout()
plt.savefig(out("03_contribucion_heatmap.png"),dpi=220,bbox_inches="tight",facecolor="white")
plt.close(); print("  03_contribucion_heatmap.png")

# Pie + barplot
dom_counts=df_ca["patron_dominante"].value_counts()
dom_order_a=[f"Patron_{p+1}" for p in range(N_PAT) if f"Patron_{p+1}" in dom_counts.index]
fig,axes=plt.subplots(1,2,figsize=(13,5))
axes[0].pie([dom_counts.get(k,0) for k in dom_order_a],
            labels=[f"P{k.split("_")[1]} (n={dom_counts.get(k,0)})" for k in dom_order_a],
            colors=[COLORS_PAT[int(k.split("_")[1])-1] for k in dom_order_a],
            autopct="%1.0f%%",startangle=90,textprops={"fontsize":10})
axes[0].set_title(f"Patrón Dominante (n={N_CASOS} casos)",fontsize=12,fontweight="bold")
means_a=df_ca[pat_cols_a].mean(); stds_a=df_ca[pat_cols_a].std()
bars_a=axes[1].bar(range(N_PAT),means_a.values,color=COLORS_PAT,
                    edgecolor="black",lw=0.8,alpha=0.85,yerr=stds_a.values,capsize=5)
axes[1].set_xticks(range(N_PAT))
axes[1].set_xticklabels([f"Patrón {p+1}" for p in range(N_PAT)],fontsize=11)
axes[1].set_ylabel("Contribución media (normalizada)",fontsize=11)
axes[1].set_title("Contribución Media ± SD por Patrón",fontsize=12,fontweight="bold")
axes[1].set_ylim(0,1.0); axes[1].grid(axis="y",alpha=0.3)
for i,(bar,val,std) in enumerate(zip(bars_a,means_a.values,stds_a.values)):
    axes[1].text(bar.get_x()+bar.get_width()/2,val+std+0.02,f"{val:.3f}",
                 ha="center",fontsize=9,color=COLORS_PAT[i],fontweight="bold")
plt.tight_layout()
plt.savefig(out("04_distribucion_patrones.png"),dpi=220,bbox_inches="tight",facecolor="white")
plt.close(); 

# Correlación features
cols_bio=[c for c in df.columns if any(x in c for x in
          ["(Tumor)","(Muscle)","(No tumor/No muscle)","Interface"])]
cols_bio=[c for c in cols_bio if df[c].dtype in [float,"float64"]]
df_cm=df.groupby("Case")[cols_bio].mean().reset_index()
df_mg=df_cm.merge(df_ca[["Case"]+pat_cols_a],on="Case")
corr_rows=[{"patron":pat,"feature":feat,"r":df_mg[[pat,feat]].corr().iloc[0,1]}
           for pat in pat_cols_a for feat in cols_bio]
df_corr=pd.DataFrame(corr_rows)
fig,axes=plt.subplots(1,N_PAT,figsize=(5*N_PAT,7))
for p,ax in enumerate(axes):
    col=COLORS_PAT[p]; pat=f"Patron_{p+1}"
    sub=df_corr[df_corr["patron"]==pat].set_index("feature")["r"]
    top=pd.concat([sub.nlargest(6),sub.nsmallest(6)]).sort_values()
    colors_b=["#C0392B" if v>0 else "#2980B9" for v in top.values]
    ax.barh(range(len(top)),top.values,color=colors_b,edgecolor="white",lw=0.5)
    ax.set_yticks(range(len(top)))
    labels=[f.replace(" (No tumor/No muscle)","\n(Stroma)").replace(" (Tumor)","\n(T)")
             .replace(" (Muscle)","\n(M)").replace("Interface Length ","Interf.") for f in top.index]
    ax.set_yticklabels(labels,fontsize=7)
    ax.axvline(0,color="black",lw=0.8)
    ax.set_title(f"Patrón {p+1}",fontsize=11,fontweight="bold",color=col)
    ax.set_xlabel("Pearson r",fontsize=9); ax.set_xlim(-0.8,0.8); ax.grid(axis="x",alpha=0.3)
    for spine in ax.spines.values():
        spine.set_edgecolor(col); spine.set_linewidth(1.5)
plt.suptitle("Correlación Features Biológicas × Patrón Tucker",fontsize=13,fontweight="bold")
plt.tight_layout()
plt.savefig(out("05_correlacion_features.png"),dpi=200,bbox_inches="tight",facecolor="white")
plt.close(); 

#  8. SUPERFIGURA Z-SCORE (para distinguir patrones)

# Calcular media por patrón dominante
df_ca["_pat_dom_num"] = df_ca["patron_dominante"].str.extract(r"(\d+)").astype(int)-1
df_merged = df.merge(df_ca[["Case","_pat_dom_num"]],on="Case",how="left")
df_merged["_pat_dom"] = df_merged["_pat_dom_num"]

cols_an=[c for c in df.columns if df[c].dtype in [float,"float64"]
         and "PC" not in c.upper() and not c.startswith("_") and c!="Case"]
df_cont = df_merged.groupby("_pat_dom")[cols_an].mean().T
df_cont.columns=[f"Patron_{i+1}" for i in df_cont.columns]

# Z-score por feature entre patrones
df_z = df_cont.apply(lambda x: (x-x.mean())/(x.std()+1e-12), axis=1)
lista_f = sorted(df_z.index, reverse=True)
pos_y = np.arange(len(lista_f))

fig,axes=plt.subplots(1,N_PAT,figsize=(22,32),sharey=True)
plt.subplots_adjust(wspace=0.02)
for p in range(N_PAT):
    ax=axes[p]; col=COLORS_PAT[p]
    vals=df_z.reindex(lista_f)[f"Patron_{p+1}"].values
    bar_colors=["#C0392B" if v>0 else "#2980B9" for v in vals]
    ax.barh(pos_y,vals,color=bar_colors,edgecolor="none",height=0.7,alpha=0.85)
    ax.axvline(0,color="black",lw=1,alpha=0.6)
    ax.set_xlim(-3,3); ax.grid(axis="x",linestyle="--",alpha=0.3)
    ax.set_title(f"PATRÓN {p+1}\n(Z-score)",fontsize=14,fontweight="bold",color=col)
    ax.set_xlabel("Desviaciones de la media",fontsize=9)
    if p==0:
        ax.set_yticks(pos_y); ax.set_yticklabels(lista_f,fontsize=6.5)
    else:
        ax.tick_params(left=False)
    for spine in ax.spines.values():
        spine.set_edgecolor(col); spine.set_linewidth(2.5)
plt.suptitle("PERFIL DIFERENCIAL POR PATRÓN TUCKER — Z-score\n"
             "Rojo = por encima de la media entre patrones | Azul = por debajo\n"
             "Permite identificar qué hace DISTINTO a cada patrón",
             fontsize=18,fontweight="bold",y=0.995)
#plt.savefig(out("SUPER_FIGURA_zscore.png"),dpi=180,bbox_inches="tight",facecolor="white")
plt.close(); 

#  9. SUPERFIGURA HEATMAP DIFERENCIAL 

# Top features más discriminantes (mayor varianza entre patrones)
varianza_entre_pat = df_z.var(axis=1).sort_values(ascending=False)
top_features = varianza_entre_pat.head(40).index.tolist()

df_z_top = df_z.loc[top_features]

fig,ax=plt.subplots(figsize=(8,14),facecolor="white")
im=ax.imshow(df_z_top.values, cmap="RdBu_r", vmin=-2.5, vmax=2.5,
             aspect="auto")
ax.set_xticks(range(N_PAT))
ax.set_xticklabels([f"Patrón {p+1}" for p in range(N_PAT)],fontsize=12,fontweight="bold")
for tick,col in zip(ax.get_xticklabels(),COLORS_PAT):
    tick.set_color(col)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features,fontsize=7.5)
ax.set_title("Top 40 features más discriminantes\n"
             "Rojo = alto en ese patrón | Azul = bajo",
             fontsize=13,fontweight="bold",pad=10)
plt.colorbar(im,ax=ax,label="Z-score",shrink=0.4)

# Separadores entre patrones
for x in range(N_PAT-1):
    ax.axvline(x+0.5,color="white",lw=2)

# Borde de color por patrón
for p in range(N_PAT):
    r_c,g_c,b_c=[int(COLORS_PAT[p][i:i+2],16)/255 for i in (1,3,5)]
    ax.add_patch(plt.Rectangle((p-0.5,-0.5),1,len(top_features),
                                fill=False,edgecolor=(r_c,g_c,b_c),lw=3,zorder=5))

plt.tight_layout()
plt.savefig(out("SUPER_FIGURA_heatmap_diferencial.png"),dpi=200,bbox_inches="tight",facecolor="white")
plt.close(); 

# Exportar CSVs
df_ca[["Case"]+pat_cols_a+["patron_dominante"]].to_csv(
    out("TUCKER_contribuciones_por_caso.csv"),index=False,sep=";",encoding="utf-8-sig")
df_corr.to_csv(out("TUCKER_correlacion_features.csv"),index=False,sep=";",encoding="utf-8-sig")


#  6. HEATMAP CONTRIBUCIONES + PIE + CORRELACIONES

print("\nGenerando figuras de análisis Tucker...")
import seaborn as sns
from PIL import ImageDraw

pat_cols_a = [f"Patron_{p+1}" for p in range(N_PAT)]
F_norm_a = best_facs[0] / (best_facs[0].max(axis=0, keepdims=True)+1e-12)
df_ca = pd.DataFrame(F_norm_a, columns=pat_cols_a)
df_ca["Case"] = casos
df_ca["patron_dominante"] = df_ca[pat_cols_a].idxmax(axis=1)

# Entropías
entropias_a = []
for p in range(N_PAT):
    pf = pattern_maps[p].flatten(); pf=pf[pf>0]; pf/=pf.sum()
    entropias_a.append(-np.sum(pf*np.log2(pf+1e-12)))

# Heatmap contribuciones
data_heat = df_ca[pat_cols_a].T
data_heat.columns = df_ca["Case"]
order_h = df_ca.assign(_dom=df_ca[pat_cols_a].idxmax(axis=1)).sort_values(
    [f"Patron_{p+1}" for p in range(N_PAT)], ascending=False).index
data_heat = data_heat[df_ca.loc[order_h, "Case"]]
fig, ax = plt.subplots(figsize=(min(N_CASOS*0.35+2, 30), 5))
sns.heatmap(data_heat, ax=ax, cmap="YlOrRd", vmin=0, vmax=1, xticklabels=True,
            yticklabels=[f"P{p+1}" for p in range(N_PAT)],
            cbar_kws={"label":"Contribución norm.", "shrink":0.6},
            linewidths=0.3, linecolor="#CCCCCC")
ax.set_xticklabels(ax.get_xticklabels(), rotation=75, ha="right", fontsize=5.5)
for label, col in zip(ax.get_yticklabels(), COLORS_PAT):
    label.set_color(col)
ax.set_title(f"Contribución de Patrones Tucker por Caso (n={N_CASOS})", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(out("03_contribucion_heatmap.png"), dpi=220, bbox_inches="tight", facecolor="white")
plt.close(); print("   03_contribucion_heatmap.png")

# Pie + barplot
dom_counts = df_ca["patron_dominante"].value_counts()
dom_order_a = [f"Patron_{p+1}" for p in range(N_PAT) if f"Patron_{p+1}" in dom_counts.index]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].pie([dom_counts.get(k,0) for k in dom_order_a],
            labels=[f"P{k.split('_')[1]} (n={dom_counts.get(k,0)})" for k in dom_order_a],
            colors=[COLORS_PAT[int(k.split("_")[1])-1] for k in dom_order_a],
            autopct="%1.0f%%", startangle=90, textprops={"fontsize":10})
axes[0].set_title(f"Patrón Dominante (n={N_CASOS} casos)", fontsize=12, fontweight="bold")
means_a = df_ca[pat_cols_a].mean(); stds_a = df_ca[pat_cols_a].std()
bars_a = axes[1].bar(range(N_PAT), means_a.values, color=COLORS_PAT,
                      edgecolor="black", lw=0.8, alpha=0.85, yerr=stds_a.values, capsize=5)
axes[1].set_xticks(range(N_PAT))
axes[1].set_xticklabels([f"Patrón {p+1}" for p in range(N_PAT)], fontsize=11)
axes[1].set_ylabel("Contribución media (normalizada)", fontsize=11)
axes[1].set_title("Contribución Media ± SD por Patrón", fontsize=12, fontweight="bold")
axes[1].set_ylim(0, 1.0); axes[1].grid(axis="y", alpha=0.3)
for i, (bar, val, std) in enumerate(zip(bars_a, means_a.values, stds_a.values)):
    axes[1].text(bar.get_x()+bar.get_width()/2, val+std+0.02, f"{val:.3f}",
                 ha="center", fontsize=9, color=COLORS_PAT[i], fontweight="bold")
plt.tight_layout()
plt.savefig(out("04_distribucion_patrones.png"), dpi=220, bbox_inches="tight", facecolor="white")
plt.close();

# Correlación features
cols_bio = [c for c in df.columns if any(x in c for x in
            ["(Tumor)", "(Muscle)", "(No tumor/No muscle)", "Interface"])]
cols_bio = [c for c in cols_bio if df[c].dtype in [float, "float64"]]
df_cm = df.groupby("Case")[cols_bio].mean().reset_index()
df_mg = df_cm.merge(df_ca[["Case"]+pat_cols_a], on="Case")
corr_rows = [{"patron":pat, "feature":feat, "r":df_mg[[pat,feat]].corr().iloc[0,1]}
             for pat in pat_cols_a for feat in cols_bio]
df_corr = pd.DataFrame(corr_rows)
fig, axes = plt.subplots(1, N_PAT, figsize=(5*N_PAT, 7))
for p, ax in enumerate(axes):
    col = COLORS_PAT[p]; pat = f"Patron_{p+1}"
    sub = df_corr[df_corr["patron"]==pat].set_index("feature")["r"]
    top = pd.concat([sub.nlargest(6), sub.nsmallest(6)]).sort_values()
    colors_b = ["#C0392B" if v>0 else "#2980B9" for v in top.values]
    ax.barh(range(len(top)), top.values, color=colors_b, edgecolor="white", lw=0.5)
    ax.set_yticks(range(len(top)))
    labels = [f.replace(" (No tumor/No muscle)", "\n(Stroma)").replace(" (Tumor)", "\n(T)")
               .replace(" (Muscle)", "\n(M)").replace("Interface Length ", "Interf.") for f in top.index]
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(f"Patrón {p+1}", fontsize=11, fontweight="bold", color=col)
    ax.set_xlabel("Pearson r", fontsize=9); ax.set_xlim(-0.8, 0.8); ax.grid(axis="x", alpha=0.3)
    for spine in ax.spines.values():
        spine.set_edgecolor(col); spine.set_linewidth(1.5)
plt.suptitle("Correlación Features Biológicas × Patrón Tucker", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(out("05_correlacion_features.png"), dpi=200, bbox_inches="tight", facecolor="white")
plt.close(); 

# CSVs
df_ca[["Case"]+pat_cols_a+["patron_dominante"]].to_csv(
    out("TUCKER_contribuciones_por_caso.csv"), index=False, sep=";", encoding="utf-8-sig")
df_corr.to_csv(out("TUCKER_correlacion_features.csv"), index=False, sep=";", encoding="utf-8-sig")
pd.DataFrame({"Patron":pat_cols_a, "Entropia_bits":[round(e,4) for e in entropias_a]}).to_csv(
    out("TUCKER_entropias.csv"), index=False, sep=";", encoding="utf-8-sig")


#  7. SUPERFIGURA: MEDIA REAL DE FEATURES POR PATRÓN

# Asignar patrón dominante del BIN a cada tile
df_ca["_pat_dom_num"] = df_ca["patron_dominante"].str.extract(r"(\d+)").astype(int) - 1
df_merged = df.merge(df_ca[["Case", "_pat_dom_num"]], on="Case", how="left")

# Variables biológicas (excluir PCs y columnas internas)
cols_an = [c for c in df.columns if df[c].dtype in [float, "float64"]
           and "PC" not in c.upper()
           and not c.startswith("_")
           and c != "Case"
           and c not in [col_x1, col_y1, col_x2, col_y2]]

# Media real por patrón: todos los tiles cuyo BIN pertenece a ese patrón
df_media = df_merged.groupby("_pat_dom")[cols_an].mean().T
df_media.columns = [f"Patron_{i+1}" for i in df_media.columns]

# Normalizar cada feature al rango [0,1] para poder comparar

df_media_norm = df_media.copy()
for feat in df_media_norm.index:
    row = df_media_norm.loc[feat]
    rng = row.max() - row.min()
    if rng > 0:
        df_media_norm.loc[feat] = (row - row.min()) / rng
    else:
        df_media_norm.loc[feat] = 0.5  # sin variación entre patrones

# Ordenar features alfabéticamente
lista_f = sorted(df_media_norm.index, reverse=True)
pos_y = np.arange(len(lista_f))

fig, axes = plt.subplots(1, N_PAT, figsize=(22, 32), sharey=True)
plt.subplots_adjust(wspace=0.02)

for p in range(N_PAT):
    ax = axes[p]; col = COLORS_PAT[p]
    vals = df_media_norm.reindex(lista_f)[f"Patron_{p+1}"].values

    # Color: más rojo = más alto en ese patrón respecto a los demás
    # más azul = más bajo
    media_entre_pat = df_media_norm.reindex(lista_f).mean(axis=1).values
    vals_rel = vals - media_entre_pat
    bar_colors = ["#C0392B" if v > 0 else "#2980B9" for v in vals_rel]

    ax.barh(pos_y, vals, color=bar_colors, edgecolor="none", height=0.7, alpha=0.88)
    ax.axvline(0.5, color="black", lw=1, alpha=0.5, linestyle="--")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_title(f"PATRÓN {p+1}\nH={entropias_a[p]:.2f} bits",
                 fontsize=14, fontweight="bold", color=col)
    ax.set_xlabel("Media normalizada [0-1]\nRojo=por encima media entre patrones | Azul=por debajo",
                  fontsize=7)
    if p == 0:
        ax.set_yticks(pos_y)
        ax.set_yticklabels(lista_f, fontsize=6.5)
    else:
        ax.tick_params(left=False)
    for spine in ax.spines.values():
        spine.set_edgecolor(col); spine.set_linewidth(2.5)

plt.suptitle("MEDIA REAL DE FEATURES POR PATRÓN TUCKER\n",
             fontsize=16, fontweight="bold", y=0.995)
#plt.savefig(out("SUPER_FIGURA_media_por_patron.png"), dpi=180, bbox_inches="tight", facecolor="white")
plt.close();# print

# 8. HEATMAP DIFERENCIAL COMPACTO
#    Top 40 features más discriminantes entre patrones

# Calcular Z-score para ver qué diferencia los patrones
df_z = df_media_norm.apply(lambda x: (x - x.mean()) / (x.std()+1e-12), axis=1)

# Top 40 features con mayor varianza entre patrones
varianza = df_z.var(axis=1).sort_values(ascending=False)
top40 = varianza.head(40).index.tolist()
df_z_top = df_z.loc[top40]

fig, ax = plt.subplots(figsize=(9, 14), facecolor="white")
im = ax.imshow(df_z_top.values, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")

ax.set_xticks(range(N_PAT))
ax.set_xticklabels([f"Patrón {p+1}" for p in range(N_PAT)], fontsize=12, fontweight="bold")
for tick, col in zip(ax.get_xticklabels(), COLORS_PAT):
    tick.set_color(col)
ax.set_yticks(range(len(top40)))
ax.set_yticklabels(top40, fontsize=7.5)

# Valores numéricos dentro de cada celda
for i in range(len(top40)):
    for j in range(N_PAT):
        val = df_z_top.values[i, j]
        ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                fontsize=5.5, color="white" if abs(val) > 1.2 else "black")

ax.set_title("Top 40 features más discriminantes entre patrones\n"
             ,
             fontsize=12, fontweight="bold", pad=10)
plt.colorbar(im, ax=ax, label="Z-score", shrink=0.4)

# Separadores y bordes de color
for x in range(N_PAT-1):
    ax.axvline(x+0.5, color="white", lw=2)
for p in range(N_PAT):
    r_c, g_c, b_c = [int(COLORS_PAT[p][i:i+2], 16)/255 for i in (1, 3, 5)]
    ax.add_patch(plt.Rectangle((p-0.5, -0.5), 1, len(top40),
                                fill=False, edgecolor=(r_c,g_c,b_c), lw=3, zorder=5))

plt.tight_layout()
plt.savefig(out("SUPER_FIGURA_heatmap_diferencial.png"), dpi=200, bbox_inches="tight", facecolor="white")
plt.close(); 


#  EXPORTAR TUCKER_MODEL.pkl
import pickle

print("\nExportando TUCKER_MODEL.pkl...")

tucker_model = {
    'best_core':      best_core,
    'best_facs':      best_facs,
    'best_err':       best_err,
    'pattern_maps':   pattern_maps,
    'bin_patron_dom': bin_patron_dom,
    'pc1_edges':      pc1_edges,
    'pc2_edges':      pc2_edges,
    'casos':          casos,
    'df_c':           df_c,   # scores Tucker normalizados por caso
}

pkl_path = out("TUCKER_MODEL.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(tucker_model, f)
print(f"   TUCKER_MODEL.pkl exportado → {pkl_path}")

#  SUPERFIGURA MEJOR PARA MEDIA REAL POR PATRÓN (

col_x1 = 'Object Info (tile) - Envelope left'
col_y1 = 'Object Info (tile) - Envelope top'
col_x2 = 'Object Info (tile) - Envelope right'
col_y2 = 'Object Info (tile) - Envelope bottom'
excluir = {col_x1, col_y1, col_x2, col_y2,
           'Case', 'PC1', 'PC2', '_b1', '_b2', '_pat_dom', '_pat_intens'}

cols_bio_v2 = [c for c in df.columns
               if df[c].dtype in [float, 'float64']
               and c not in excluir
               and 'PC' not in c.upper()
               and not c.startswith('_')]

PAT_NOMBRES_V2 = ['P1 Tumor compacto', 'P2 Musculo+chispitas',
                  'P3 Interfaz estromal', 'P4 Invasion infiltrativa', 'P5 Tumor masivo']

# Media real de cada feature por patron (usando _pat_dom del BIN)
df_media_v2 = df.groupby('_pat_dom')[cols_bio_v2].mean().T
df_media_v2.columns = [f'Patron_{i+1}' for i in df_media_v2.columns]

# Normalizar min-max entre patrones: 0=minimo, 1=maximo
df_norm_v2 = df_media_v2.copy()
for feat in df_norm_v2.index:
    row = df_norm_v2.loc[feat]
    rng = row.max() - row.min()
    df_norm_v2.loc[feat] = (row - row.min()) / rng if rng > 1e-10 else 0.5

# Agrupar por tejido
grupos_v2 = {
    'MUSCULO':  [c for c in cols_bio_v2 if '(Muscle)' in c],
    'TUMOR':    [c for c in cols_bio_v2 if '(Tumor)' in c],
    'ESTROMA':  [c for c in cols_bio_v2 if '(No tumor/No muscle)' in c],
    'INTERFAZ': [c for c in cols_bio_v2 if 'Interface' in c],
}

def nombre_corto_v2(f):
    for rep, sust in [(' (No tumor/No muscle)', ''), (' (Muscle)', ''), (' (Tumor)', ''),
                      ('Object Info (tile) - ', ''), ('Interface Length ', 'Interf. '),
                      ('Area_div_', 'A/'), ('Entropy 32bins ', 'Entr. '), ('Intensity', 'Int.')]:
        f = f.replace(rep, sust)
    return f

orden_v2 = []; etiquetas_v2 = []; sep_v2 = {}; pos_v2 = 0
for gnom, feats in grupos_v2.items():
    if not feats: continue
    feats_s = sorted(feats)
    sep_v2[gnom] = (pos_v2, pos_v2 + len(feats_s))
    for f in feats_s:
        orden_v2.append(f); etiquetas_v2.append(nombre_corto_v2(f))
    pos_v2 += len(feats_s)

pos_y_v2 = np.arange(len(orden_v2))
COLORES_GRUPO_V2 = ['#F8F8F8', '#EFEFEF', '#F8F8F8', '#EFEFEF']

fig, axes = plt.subplots(1, N_PAT,
                          figsize=(24, max(18, len(orden_v2)*0.25 + 2)),
                          sharey=True, facecolor='white')
plt.subplots_adjust(wspace=0.02, left=0.18, right=0.97, top=0.94, bottom=0.04)

for p in range(N_PAT):
    ax = axes[p]; col = COLORS_PAT[p]
    pat_col = f'Patron_{p+1}'

    if pat_col not in df_norm_v2.columns:
        ax.set_visible(False); continue

    vals = df_norm_v2.reindex(orden_v2)[pat_col].values

    # Franjas de fondo alternadas por grupo
    g_idx = 0
    for gnom, (g_ini, g_fin) in sep_v2.items():
        ax.axhspan(g_ini - 0.5, g_fin - 0.5,
                   facecolor=COLORES_GRUPO_V2[g_idx % 4], alpha=0.5, zorder=0)
        g_idx += 1

    # Barras del color del patron (sin rojo/azul, solo el color del patron)
    ax.barh(pos_y_v2, vals, color=col, edgecolor='none',
            height=0.75, alpha=0.88, zorder=2)

    # Linea de referencia en 0.5
    ax.axvline(0.5, color='gray', lw=0.8, alpha=0.6, linestyle='--', zorder=3)

    # Separadores y etiquetas de grupo
    for gnom, (g_ini, g_fin) in sep_v2.items():
        ax.axhline(g_ini - 0.5, color='black', lw=1.5, alpha=0.7, zorder=4)
        if p == 0:
            ax.text(-0.02, (g_ini + g_fin) / 2 - 0.5, gnom,
                    ha='right', va='center', fontsize=8, fontweight='bold',
                    color='#333333', transform=ax.get_yaxis_transform())

    n_tiles = (df['_pat_dom'] == p).sum()
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'], fontsize=7)
    ax.grid(axis='x', linestyle='--', alpha=0.25, zorder=1)
    ax.set_title(f'{PAT_NOMBRES_V2[p]}\n({n_tiles:,} tiles)',
                 fontsize=10, fontweight='bold', color=col, pad=6)
    ax.set_xlabel('[0=minimo, 1=maximo entre patrones]', fontsize=7)

    for spine in ax.spines.values():
        spine.set_edgecolor(col); spine.set_linewidth(2.5)

    if p == 0:
        ax.set_yticks(pos_y_v2)
        ax.set_yticklabels(etiquetas_v2, fontsize=6.5)
    else:
        ax.tick_params(left=False)

plt.suptitle('PERFIL MORFOLOGICO POR PATRON TUCKER\n',
             
             fontsize=13, fontweight='bold', y=0.98)
plt.savefig(out('SUPER_FIGURA_media_patron_v2.png'),
            dpi=180, bbox_inches='tight', facecolor='white')
plt.close()



# EXPORTAR CSV DE TILES CON PATRON DOMINANTE


cols_export_tiles = [
    'Case',
    col_x1, col_y1, col_x2, col_y2,
    '_b1', '_b2', '_pat_dom', '_pat_intens'
]
# Añadir PCs si están disponibles
cols_export_tiles += [c for c in ['PC1','PC2'] if c in df.columns]

df[cols_export_tiles].to_csv(
    out('TUCKER_tiles_patron_dom.csv'),
    index=False, sep=';', encoding='utf-8-sig', decimal=','
)
print(f"   TUCKER_tiles_patron_dom.csv ({len(df):,} filas)")




########MEJORAS########


# MEJORA 1: SUPER_FIGURA_media_patron_v2 con etiquetas verticales

fig, axes = plt.subplots(1, N_PAT,
                          figsize=(24, max(18, len(orden_v2)*0.25 + 2)),
                          sharey=True, facecolor='white')
plt.subplots_adjust(wspace=0.02, left=0.22, right=0.97, top=0.94, bottom=0.04)

for p in range(N_PAT):
    ax = axes[p]; col = COLORS_PAT[p]
    pat_col = f'Patron_{p+1}'

    if pat_col not in df_norm_v2.columns:
        ax.set_visible(False); continue

    vals = df_norm_v2.reindex(orden_v2)[pat_col].values

    # Franjas de fondo alternadas por grupo
    g_idx = 0
    for gnom, (g_ini, g_fin) in sep_v2.items():
        ax.axhspan(g_ini - 0.5, g_fin - 0.5,
                   facecolor=COLORES_GRUPO_V2[g_idx % 4], alpha=0.5, zorder=0)
        g_idx += 1

    ax.barh(pos_y_v2, vals, color=col, edgecolor='none',
            height=0.75, alpha=0.88, zorder=2)
    ax.axvline(0.5, color='gray', lw=0.8, alpha=0.6, linestyle='--', zorder=3)

    # Separadores y etiquetas de grupo — ahora en VERTICAL y más a la izquierda
    for gnom, (g_ini, g_fin) in sep_v2.items():
        ax.axhline(g_ini - 0.5, color='black', lw=1.5, alpha=0.7, zorder=4)
        if p == 0:
            ax.text(-0.25,                          # ← más a la izquierda
                    (g_ini + g_fin) / 2 - 0.5,
                    gnom,
                    ha='center', va='center',
                    fontsize=8, fontweight='bold',
                    color='#333333',
                    rotation=90,                    # ← vertical
                    transform=ax.get_yaxis_transform())

    n_tiles = (df['_pat_dom'] == p).sum()
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'], fontsize=7)
    ax.grid(axis='x', linestyle='--', alpha=0.25, zorder=1)
    ax.set_title(f'{PAT_NOMBRES_V2[p]}\n({n_tiles:,} tiles)',
                 fontsize=10, fontweight='bold', color=col, pad=6)
    ax.set_xlabel('[0=minimo, 1=maximo entre patrones]', fontsize=7)

    for spine in ax.spines.values():
        spine.set_edgecolor(col); spine.set_linewidth(2.5)

    if p == 0:
        ax.set_yticks(pos_y_v2)
        ax.set_yticklabels(etiquetas_v2, fontsize=6.5)
    else:
        ax.tick_params(left=False)

plt.suptitle('PERFIL MORFOLOGICO POR PATRON TUCKER\n',
             fontsize=13, fontweight='bold', y=0.98)
plt.savefig(out('SUPER_FIGURA_media_patron_v2.png'),
            dpi=180, bbox_inches='tight', facecolor='white')
plt.close()




df_export_features = pd.DataFrame(index=orden_v2)
df_export_features.index.name = 'Feature'

# Etiqueta corta
df_export_features['Etiqueta_corta'] = etiquetas_v2
df_export_features['Grupo'] = [
    next((g for g, (gi, gf) in sep_v2.items() if gi <= i < gf), 'OTRO')
    for i in range(len(orden_v2))
]

for p in range(N_PAT):
    pat_col = f'Patron_{p+1}'
    df_export_features[f'Media_real_P{p+1}']  = df_media_v2.reindex(orden_v2)[pat_col].values
    df_export_features[f'Media_norm_P{p+1}']  = df_norm_v2.reindex(orden_v2)[pat_col].values

df_export_features.to_csv(
    out('features_medias_por_patron.csv'),
    sep=';', encoding='utf-8-sig', decimal=',', index=True
)
print(f"   features_medias_por_patron.csv ({len(df_export_features)} features)")


# MEJORA 2: TUCKER_contribuciones_por_caso.xlsx


# Hoja 1: contribuciones normalizadas
df_hoja1 = df_ca[['Case'] + pat_cols_a + ['patron_dominante']].copy()

# Hoja 2: tiles por patrón dominante por caso
# _pat_dom está en df (a nivel tile), patron dominante del BIN
registros = []
for caso_i in casos:
    df_tiles_caso = df[df['Case'] == caso_i]
    fila = {'Case': caso_i, 'Total_tiles': len(df_tiles_caso)}
    for p in range(N_PAT):
        fila[f'Tiles_Patron_{p+1}'] = int((df_tiles_caso['_pat_dom'] == p).sum())
        fila[f'Pct_Patron_{p+1}']   = round(
            (df_tiles_caso['_pat_dom'] == p).sum() / max(len(df_tiles_caso), 1) * 100, 1)
    fila['Patron_dominante_tiles'] = f"Patron_{int(df_tiles_caso['_pat_dom'].value_counts().idxmax())+1}"
    registros.append(fila)

df_hoja2 = pd.DataFrame(registros)

xlsx_path = out('TUCKER_contribuciones_por_caso.xlsx')
with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
    df_hoja1.to_excel(writer, sheet_name='Contribuciones_Tucker', index=False)
    df_hoja2.to_excel(writer, sheet_name='Tiles_por_patron',      index=False)

    # Ajustar ancho de columnas automáticamente en ambas hojas
    for sheet_name in writer.sheets:
        ws = writer.sheets[sheet_name]
        for col in ws.columns:
            max_len = max(len(str(cell.value)) if cell.value else 0 for cell in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 30)



# para ejecutar #py -3.12 c:/Users/carme/OneDrive/Escritorio/M.UCM/TFM/1700MICRAS/CENTROIDE/FEATURE_ENGINEERING/TUCKER/tucker_solapamiento_fondo_eda2.py