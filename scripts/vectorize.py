# -*- coding: utf-8 -*-
"""
Растр -> вектор: трассировка плана 1-го этажа и сборка DXF (масштаб 1:200, мм).
Пайплайн:
  1) извлечение растра из PDF (PyMuPDF), склейка полос;
  2) маски: чёрная графика (<150) и серая заливка стен (~199-200) без гало;
  3) суперсэмплинг x2 + potrace (obratrace) -> контуры;
  4) DXF R2018: слои РАСТР_ПОДЛОЖКА / СТЕНЫ_ЗАЛИВКА / СТЕНЫ_КОНТУР / ПЛАН_ГРАФИКА.
"""
import io, math, sys

import numpy as np
import cv2
import fitz  # PyMuPDF
from PIL import Image
import potrace
import ezdxf
from ezdxf.enums import TextEntityAlignment

PDF = "План 1 этажа.pdf"
OUT_DXF = "План 1 этажа.dxf"
IMG_FOR_DXF = "plan_1_floor_raster.png"  # ASCII-имя: совместимость с DWG/R2000

DARK_T = 150          # порог чёрной графики
GREY_LO, GREY_HI = 165, 232   # коридор серой заливки стен
UP = 2                # коэффициент суперсэмплинга
SCALE_DENOM = 200     # масштаб по штампу 1:200

# ---------- 1. растр из PDF ----------
doc = fitz.open(PDF)
page = doc[0]
strips = []
for img in page.get_images(full=True):
    xref = img[0]
    rect = page.get_image_rects(xref)[0]
    pil = Image.open(io.BytesIO(doc.extract_image(xref)["image"])).convert("L")
    strips.append((rect.y0, rect.x0, pil))
strips.sort(key=lambda t: t[0])
W = strips[0][2].width
H = sum(s[2].height for s in strips)
full = Image.new("L", (W, H), 255)
y = 0
for _, _, s in strips:
    full.paste(s, (0, y)); y += s.height

img_rects = [page.get_image_rects(img[0])[0] for img in page.get_images(full=True)]
img_left_pt = min(r.x0 for r in img_rects)
img_top_pt = min(r.y0 for r in img_rects)
img_w_pt = max(r.x1 for r in img_rects) - img_left_pt
img_h_pt = max(r.y1 for r in img_rects) - img_top_pt

MM_PER_PT = 25.4 / 72.0
mm_per_px_x = img_w_pt / W * MM_PER_PT * SCALE_DENOM
mm_per_px_y = img_h_pt / H * MM_PER_PT * SCALE_DENOM
origin_x_mm = img_left_pt * MM_PER_PT * SCALE_DENOM
origin_y_mm = img_top_pt * MM_PER_PT * SCALE_DENOM   # от верха листа
print(f"растр {W}x{H}px; {mm_per_px_x:.4f}x{mm_per_px_y:.4f} мм/px (натура)")

full.save("work/plan_raw.png")
full.save(IMG_FOR_DXF)  # подложка рядом с DXF
arr = np.array(full)

# ---------- 2. маски ----------
dark = (arr < DARK_T).astype(np.uint8)
grey = ((arr >= GREY_LO) & (arr <= GREY_HI)).astype(np.uint8)
# убрать гало антиалиасинга вокруг чёрных линий из серой маски
kernel = np.ones((3, 3), np.uint8)
dark_dil = cv2.dilate(dark, kernel, iterations=2)
grey[dark_dil > 0] = 0
# закрыть мелкие дырки в стенах, убрать одиночные пиксели
grey = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, kernel, iterations=1)
grey = cv2.morphologyEx(grey, cv2.MORPH_OPEN, kernel, iterations=1)

print("чёрных px:", int(dark.sum()), "серых px:", int(grey.sum()))

# ---------- 3. potrace ----------
def trace(mask01, turd):
    """маска 0/1 (1 = графика) -> плоский список замкнутых контуров
    (внешние границы и границы дырок). pts в пикселях картинки xUP."""
    big = cv2.resize(mask01, None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)
    img255 = np.where(big > 0.5, 0, 255).astype(np.uint8)  # графика = чёрный (0)
    plist = potrace.Bitmap(img255).trace(turdsize=turd, alphamax=1.0,
                                         opticurve=True, opttolerance=0.3)

    def flatten(path):
        xy = lambda p: (float(p.x), float(p.y))
        pts = [xy(path.start_point)]
        for seg in path:
            if seg.is_corner:
                pts.append(xy(seg.c)); pts.append(xy(seg.end_point))
            else:
                p0 = np.array(pts[-1]); p1 = np.array(xy(seg.c1))
                p2 = np.array(xy(seg.c2)); p3 = np.array(xy(seg.end_point))
                for t in np.linspace(0.1, 1.0, 10):
                    p = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 \
                        + 3 * (1 - t) * t * t * p2 + t ** 3 * p3
                    pts.append((float(p[0]), float(p[1])))
        return pts

    out = []
    for path in plist:
        pts = flatten(path)
        if len(pts) >= 3:
            out.append(pts)
    return out

def rdp(pts, eps):
    """Ramer–Douglas–Peucker для ЗАМКНУТОГО контура (итеративно)."""
    n = len(pts)
    if n < 4:
        return pts
    arrp = np.asarray(pts, dtype=float)
    # убрать подряд идущие дубликаты (угловые сегменты potrace такие дают)
    keep0 = np.ones(n, bool)
    keep0[1:] = (np.abs(np.diff(arrp[:, 0])) + np.abs(np.diff(arrp[:, 1]))) > 1e-9
    arrp = arrp[keep0]
    n = len(arrp)
    if n < 4:
        return [tuple(p) for p in arrp]
    # разрываем кольцо в точке, самой удалённой от центроида
    c = arrp.mean(axis=0)
    i0 = int(np.argmax(((arrp - c) ** 2).sum(axis=1)))
    a2 = np.vstack([arrp[i0:], arrp[:i0]])
    m = len(a2)
    keep = np.zeros(m, bool); keep[0] = keep[-1] = True
    stack = [(0, m - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        p0, p1 = a2[i], a2[j]
        v = p1 - p0
        nv = math.hypot(v[0], v[1])
        rel = a2[i + 1:j] - p0
        if nv < 1e-12:
            d = np.hypot(rel[:, 0], rel[:, 1])
        else:
            d = np.abs(rel[:, 0] * v[1] - rel[:, 1] * v[0]) / nv
        k = int(np.argmax(d)) + i + 1
        if d[k - i - 1] > eps:
            keep[k] = True
            stack.append((i, k)); stack.append((k, j))
    return [tuple(p) for p in a2[keep]]

import os, pickle
CACHE = "work/trace_cache.pkl"
if os.path.exists(CACHE):
    with open(CACHE, "rb") as f:
        dark_paths, grey_paths = pickle.load(f)
    print("кэш трассировки загружен")
else:
    print("трассировка чёрного слоя...")
    dark_paths = trace(dark, 10)
    print("  контуров:", len(dark_paths))
    print("трассировка серого слоя...")
    grey_paths = trace(grey, 200)
    print("  контуров:", len(grey_paths))
    with open(CACHE, "wb") as f:
        pickle.dump((dark_paths, grey_paths), f)

# ---------- 4. DXF ----------
docx = ezdxf.new("R2018", setup=True)
msp = docx.modelspace()
docx.header["$INSUNITS"] = 4  # мм
# ASCII-имена слоёв: DWG R2000 держит только однобайтовую кодировку
for name, color in [("A-RASTER-UNDERLAY", 8), ("A-WALLS-FILL", 9),
                    ("A-WALLS-OUTLINE", 8), ("A-PLAN-GRAPHICS", 7)]:
    docx.layers.add(name, color=color)

def to_mm(px, py):
    """px картинки (y вниз) -> реальные мм (y вверх), с учётом xUP и полей листа."""
    x = origin_x_mm + (px / UP) * mm_per_px_x
    y = origin_y_mm + ((H - py / UP)) * mm_per_px_y
    return x, y

def clean(pts, eps_px):
    p = rdp(pts, eps_px)
    return [to_mm(x, y) for x, y in p]

# серый: ОДНА штриховка со всеми петлями (стиль «обычный» = чередование
# чётные/нечётные области -> дырки в стенах остаются незалитыми) + контуры
grey_mm = [p for p in (clean(pts, 0.6) for pts in grey_paths) if len(p) >= 3]
h = msp.add_hatch(dxfattribs={"layer": "A-WALLS-FILL"})
h.set_solid_fill(color=9)
h.dxf.hatch_style = 0  # normal / odd parity
for mp in grey_mm:
    h.paths.add_polyline_path(mp, is_closed=True)
for mp in grey_mm:
    msp.add_lwpolyline(mp, close=True, dxfattribs={"layer": "A-WALLS-OUTLINE", "color": 8})
print("контуров стен:", len(grey_mm))

# чёрный: контуры (внешние границы и границы дырок — всё рисуем)
n_p = 0
for pts in dark_paths:
    mp = clean(pts, 0.5)
    if len(mp) < 3:
        continue
    msp.add_lwpolyline(mp, close=True, dxfattribs={"layer": "A-PLAN-GRAPHICS"})
    n_p += 1
print("контуров графики:", n_p)

# растровая подложка для визуального контроля (сохраняет «ничего не потерялось»)
img_def = docx.add_image_def(filename=IMG_FOR_DXF, size_in_pixel=(W, H))
img_h_mm = H * mm_per_px_y
img_w_mm = W * mm_per_px_x
ins = msp.add_image(insert=(origin_x_mm, origin_y_mm),
                    size_in_units=(img_w_mm, img_h_mm),
                    image_def=img_def, rotation=0,
                    dxfattribs={"layer": "A-RASTER-UNDERLAY"})
ins.dxf.flags = 9  # show + transparency on

# рамка листа по границам страницы PDF — на сервисном слое
pw = page.rect.width * MM_PER_PT * SCALE_DENOM
ph = page.rect.height * MM_PER_PT * SCALE_DENOM
docx.layers.add("SHEET-FRAME", color=1)
msp.add_lwpolyline([(0, 0), (pw, 0), (pw, ph), (0, ph)], close=True,
                   dxfattribs={"layer": "SHEET-FRAME"})

docx.saveas(OUT_DXF)
print("сохранено:", OUT_DXF)
