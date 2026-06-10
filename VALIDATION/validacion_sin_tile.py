import struct
import json
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os
import csv
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

# SIN TILE

# 1. CONFIGURACIÓN 
file_path = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\mcalvo\Validation\annotations_model\B17-11302.mld"
geojson_path = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\mcalvo\Validation\annotations_pathologist\B17-11302.czi - ScanRegion0.geojson"
pixel_size_mm = 0.0001723

ID_TUMOR = 2
ID_MUSCLE = 5

model_by_class = {"Tumor": [], "Musculo": []}
gt_by_class = {"Tumor": [], "Musculo": []}

all_metrics_rows = [] 

# 2. EXTRACCIÓN DEL MODELO (MLD) 
with open(file_path, "rb") as f:
    f.read(4)
    version = struct.unpack("<i", f.read(4))[0]
    n_layers = struct.unpack("<i", f.read(4))[0]

    for _ in range(n_layers):
        header = f.read(69)
        if len(header) < 69: break
        layer_name = header[:64].decode("ascii", errors="ignore").strip("\x00")
        n_objects_total = struct.unpack("<i", header[65:69])[0]
        print("Layer detectada:", layer_name)
        print("Objetos en layer:", n_objects_total)

        objects_read_in_layer = 0
        while objects_read_in_layer < n_objects_total:
            size_raw = f.read(4)
            if not size_raw: break
            buffer_size = struct.unpack("<i", size_raw)[0]
            buffer = f.read(buffer_size)
            offset = 0

            while offset < buffer_size:
                try:
                    shape = buffer[offset]; offset += 1
                    obj_type = buffer[offset]; offset += 1

                    if shape not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                        break

                    points = []
                    if shape in [0, 8]:
                        n_points = struct.unpack("<i", buffer[offset:offset+4])[0]
                        offset += 4
                        for _ in range(n_points):
                            x = struct.unpack("<f", buffer[offset:offset+4])[0]
                            y = struct.unpack("<f", buffer[offset+4:offset+8])[0]
                            points.append((x, y))
                            offset += 8
                        if points: points.append(points[0])
                    elif shape in [1, 2, 5, 6, 7]:
                        offset += 4 + (2*8)
                        if shape in [1, 5]: offset += 24
                        elif shape in [2, 6]: offset += 16
                    elif shape in [3, 4, 9]:
                        n_pts = struct.unpack("<i", buffer[offset:offset+4])[0] if shape == 3 else 2
                        offset += 4 + (n_pts*8) if shape == 3 else 16

                    for _ in range(2):
                        while offset < len(buffer) and buffer[offset] != 0:
                            offset += 1
                        offset += 1

                    if layer_name in ["Label", "ROI"] and len(points) >= 4:
                        poly = Polygon(points)
                        if obj_type == ID_TUMOR:
                            model_by_class["Tumor"].append(poly)
                        elif obj_type == ID_MUSCLE:
                            model_by_class["Musculo"].append(poly)

                    objects_read_in_layer += 1
                except Exception:
                    break

# 3. EXTRACCIÓN DEL PATÓLOGO (GEOJSON) 
with open(geojson_path) as f:
    data = json.load(f)

for feature in data["features"]:
    props = feature.get("properties", {})
    classification = props.get("classification")
    if classification:
        c_name = classification.get("name")
        if c_name == "Muscle": c_name = "Musculo"

        if c_name in gt_by_class:
            geom = feature["geometry"]
            if geom["type"] == "Polygon":
                all_coords = [geom["coordinates"]]
            elif geom["type"] == "MultiPolygon":
                all_coords = geom["coordinates"]
            else: continue

            for poly_coords in all_coords:
                pts_clean = [(float(p[0]), float(p[1])) for p in poly_coords[0]]
                if len(pts_clean) >= 3:
                    p = Polygon(pts_clean)
                    if not p.is_valid: p = p.buffer(0)
                    gt_by_class[c_name].append(p)

# 4. ALINEACIÓN 
def get_real_info(path):
    with open(path, "rb") as f:
        content = f.read()
    tag = b"[ImageInfo]"
    idx = content.find(tag)
    ptr = idx + len(tag) + 1
    length = struct.unpack("<Q", content[ptr:ptr+8])[0]
    xml = content[ptr+8 : ptr+8+length].decode("utf-8")
    root = ET.fromstring(xml)
    fov = root.find("FOV")
    return {"l": float(fov.find("Left").text), "t": float(fov.find("Top").text)}

img_info = get_real_info(file_path)

# 5. EVALUACIÓN 
def evaluate(target):
    global all_metrics_rows

    print(f"\nANÁLISIS DE CLASE: {target.upper()}")
    m_list = model_by_class.get(target, [])
    g_list = gt_by_class.get(target, [])

    if not m_list or not g_list:
        print(f"Sin datos suficientes. Modelo: {len(m_list)} | Patólogo: {len(g_list)}")
        return

    m_mask = unary_union([p.buffer(0) for p in m_list]).buffer(0)

    g_polys_mm = []
    for p in g_list:
        scaled = [(pt[0]*pixel_size_mm + img_info["l"], img_info["t"] - pt[1]*pixel_size_mm)
                  for pt in p.exterior.coords]
        poly_mm = Polygon(scaled)
        if not poly_mm.is_valid: poly_mm = poly_mm.buffer(0)
        g_polys_mm.append(poly_mm)

    g_mask = unary_union([p.buffer(0) for p in g_polys_mm]).buffer(0)

    # Métricas Globales
    tp = m_mask.intersection(g_mask).area
    dice_slide = (2 * tp) / (m_mask.area + g_mask.area)
    recall = tp / g_mask.area
    print(f"--- Nivel Slide (Global) ---")
    print(f"Dice Global: {dice_slide:.4f} | Recall: {recall:.4f}")

    # Nivel Local ROI
    roi = box(*g_mask.bounds).buffer(0.1)
    m_local = m_mask.intersection(roi)
    tp_l = m_local.intersection(g_mask).area
    dice_local = (2 * tp_l) / (m_local.area + g_mask.area)
    prec_local = tp_l / m_local.area if m_local.area > 0 else 0  # ← PRECISION

    print(f"--- Nivel Local (ROI) ---")
    print(f"Dice Local: {dice_local:.4f} | Precision Local: {prec_local:.4f}")

    tp_area = tp_l
    fp_area = m_local.area - tp_area
    fn_area = g_mask.area - tp_area
    print(f"--- Matriz de Confusión (Áreas en mm^2) ---")
    print(f"Verdaderos Positivos (TP): {tp_area:.4f}")
    print(f"Falsos Positivos (FP): {fp_area:.4f}")
    print(f"Falsos Negativos (FN): {fn_area:.4f}")

    
    all_metrics_rows.append({
        "Imagen": os.path.basename(file_path),
        "Clase": target,
        "Dice": round(dice_slide, 4),
        "Recall": round(recall, 4),
        "Precision": round(prec_local, 4)
    })

#  6. VISUALIZACIÓN
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
classes_to_eval = ["Tumor", "Musculo"]

def fill_geometry(ax, geom, color, label, alpha=0.7):
    if geom.is_empty: return
    parts = geom.geoms if hasattr(geom, 'geoms') else [geom]
    label_added = False
    for part in parts:
        x, y = part.exterior.xy
        current_label = label if not label_added else ""
        ax.fill(x, y, color=color, alpha=alpha, label=current_label)
        label_added = True

for i, cls in enumerate(classes_to_eval):
    evaluate(cls)
    ax = axes[i]
    m_list = model_by_class.get(cls, [])
    g_list = gt_by_class.get(cls, [])

    if m_list and g_list:
        m_mask = unary_union([p.buffer(0) for p in m_list]).buffer(0)

        g_polys_mm = []
        for p in g_list:
            scaled = [(pt[0]*pixel_size_mm + img_info["l"], img_info["t"] - pt[1]*pixel_size_mm)
                      for pt in p.exterior.coords]
            poly_mm = Polygon(scaled)
            if not poly_mm.is_valid: poly_mm = poly_mm.buffer(0)
            g_polys_mm.append(poly_mm)
        g_mask = unary_union([p.buffer(0) for p in g_polys_mm]).buffer(0)

        intersection = m_mask.intersection(g_mask)
        only_model = m_mask.difference(g_mask)
        only_gt = g_mask.difference(m_mask)

        fill_geometry(ax, only_model, color='#2ecc71', label='Modelo', alpha=0.7)
        fill_geometry(ax, only_gt, color='#e74c3c', label='Patólogo', alpha=0.7)
        fill_geometry(ax, intersection, color='#9b59b6', label='Solapamiento (Acierto)', alpha=1.0)

        ax.set_title(f"Análisis de Precisión Espacial: {cls}")
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize='small')
    else:
        ax.set_title(f"Clase {cls}: Sin datos suficientes")

plt.tight_layout()
plt.show()


for cls in classes_to_eval:
    m_list = model_by_class[cls]
    g_list = gt_by_class[cls]

    if m_list and g_list:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for p in g_list:
            p_clean = p if p.is_valid else p.buffer(0)
            sx = [pt[0]*pixel_size_mm + img_info["l"] for pt in p_clean.exterior.coords]
            sy = [img_info["t"] - pt[1]*pixel_size_mm for pt in p_clean.exterior.coords]
            ax1.fill(sx, sy, alpha=0.6, fc='blue', ec='darkblue')
        ax1.set_title(f"Patólogo (GT): {cls}")
        ax1.set_aspect('equal')

        for poly in m_list:
            poly_clean = poly if poly.is_valid else poly.buffer(0)
            x, y = poly_clean.exterior.xy
            ax2.fill(x, y, alpha=0.5, fc='red', ec='darkred')
        ax2.set_title(f"Modelo: {cls}")
        ax2.set_aspect('equal')

        plt.suptitle(f"Comparativa Independiente - Clase {cls}")
        plt.tight_layout()
        plt.show()

# 8. GUARDADO EN CSV 
csv_path = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\VALIDATION\metricas_globales.csv"

file_exists = os.path.isfile(csv_path)

with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Imagen", "Clase", "Dice", "Recall", "Precision"])

    if not file_exists:
        writer.writeheader()

    writer.writerows(all_metrics_rows)

print(f"\nSe añadieron {len(all_metrics_rows)} métricas a {csv_path}")
