# -*- coding: utf-8 -*-
"""
Растр -> вектор для «План 1 этажа.pdf».

Алгоритм:
  1. Встроенные JPEG-полосы из PDF склеиваются в одну картинку в полном
     исходном разрешении (без пересэмплирования), привязка пикселей к
     бумажным координатам берётся из размещения полос в PDF.
  2. Картинка режется на цветные слои: красные стены, синие надписи,
     чёрный/серый чертёж (текст, размеры, лестницы, колонны, штамп).
  3. Каждый слой трассируется: компонент -> один HATCH (сплошная заливка,
     островки/дырки учитываются) + LWPOLYLINE границ для привязок.
     Текст и штамп становятся кривыми: визуально не теряется НИЧЕГО,
     даже нераспознанные надписи.
  4. DXF (AutoCAD R2010), единицы — миллиметры реального здания:
     лист начерчен в М 1:200, значит 1 pt бумаги = 25.4/72 мм бумаги * 200.
  5. Оригинальный растр кладётся в DXF как подложка (слой RASTER_UNDERLAY)
     для сверки и докалибровки.

Запуск:  python3 tools/raster2dxf.py "План 1 этажа.pdf"
"""
import os
import sys

import cv2
import numpy as np
import pymupdf
import ezdxf
from PIL import Image

# ---------------- параметры ----------------
UPSCALE = 2             # апскейл перед трассировкой (гладкие края)
EPSILON = 1.2           # упрощение контуров, px (в апскейл-пространстве)
MIN_AREA = 4.0          # отсев пыли: площадь контура >=, px^2 (апскейл)
MIN_PERIM = 6.0         # и периметр >=, px (апскейл)
BIG_AREA = 15000.0      # «большой» компонент (стены и пр.) — заливку для него
                        # трассуем по слегка эродированной маске (снимает
                        # защипы самокасания на T-стыках => валидный HATCH)
PRINT_SCALE = 200.0     # лист начерчен в М 1:200 (см. штамп)
DXF_VERSION = "R2010"
# цвета заливок (TrueColor) и слоёв
FILL_WALLS = (241, 183, 183)   # розовая заливка стен как в оригинале
FILL_BLACK = (40, 40, 40)
FILL_BLUE = (0, 0, 205)
# -------------------------------------------


def extract_page_image(pdf_path):
    """Склейка растровых полос -> (PIL image, img_rect_pt, page_rect_pt)."""
    doc = pymupdf.open(pdf_path)
    page = doc[0]
    infos = sorted(page.get_image_info(xrefs=True), key=lambda i: i["bbox"][1])
    if not infos:
        raise RuntimeError("В PDF нет растровых изображений — он, похоже, уже векторный.")
    strips = []
    for info in infos:
        pix = pymupdf.Pixmap(doc, info["xref"])
        if pix.alpha or (pix.colorspace and pix.colorspace.n > 3):
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        strips.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    w = max(im.width for im in strips)
    h = sum(im.height for im in strips)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    y = 0
    for im in strips:
        canvas.paste(im, (0, y))
        y += im.height
    rect = None
    for info in infos:
        r = pymupdf.Rect(info["bbox"])
        rect = r if rect is None else (rect | r)
    return canvas, rect, page.rect


def segment_colors(img_rgb):
    r = img_rgb[:, :, 0].astype(np.int16)
    g = img_rgb[:, :, 1].astype(np.int16)
    b = img_rgb[:, :, 2].astype(np.int16)
    v = np.maximum(np.maximum(r, g), b)
    red = ((r > g + 25) & (r > b + 25) & (r > 90)).astype(np.uint8) * 255
    blue = ((b > r + 20) & (b > g + 20) & (b > 80)).astype(np.uint8) * 255
    dark = ((v < 235) & (red == 0) & (blue == 0)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    for m in (red, blue, dark):
        m[:] = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return {"walls": red, "blue": blue, "dark": dark}


def upscale_mask(m):
    up = cv2.resize(m, (m.shape[1] * UPSCALE, m.shape[0] * UPSCALE),
                    interpolation=cv2.INTER_CUBIC)
    return cv2.threshold(up, 128, 255, cv2.THRESH_BINARY)[1]


def loop_fill_ratio(cnt):
    """Доля реальной заливки контура (fillPoly) к площади по шнуровке.
    Для вырожденных «серпантинных» контуров тонких линий (вниз-вверх по
    кромкам) чётно-нечётная заливка почти пуста -> такие петли в HATCH
    не включаем, они останутся полилиниями."""
    a = abs(cv2.contourArea(cnt))
    if a < 1:
        return 0.0
    x, y, w, h = cv2.boundingRect(cnt)
    tmp = np.zeros((h + 6, w + 6), np.uint8)
    shifted = cnt - np.array([[[x - 3, y - 3]]])
    cv2.fillPoly(tmp, [shifted], 255)
    return float((tmp > 0).sum()) / a


# «простота» контура по отношению заливки к площади шнуровки.
# Дискретное раздувание fillPoly ~ полупериметр: ratio ~= 1 + P/(2A),
# поэтому верхняя граница считается от геометрии петли (+0.12 запаса).
# ratio < LOW — вырожденный серпантин тонкой линии (заливать криво).
RATIO_LOW = 0.60
EPS_LADDER = (1.2, 0.8, 0.5, 0.3, 0.15, 0.08)


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _proper_cross(a, b, c, d):
    """Строгое пересечение отрезков ab и cd (касания/общие точки — не считаем)."""
    o1, o2 = _orient(a, b, c), _orient(a, b, d)
    o3, o4 = _orient(c, d, a), _orient(c, d, b)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def has_self_intersection(pts):
    """Проверка замкнутой ломаной на строгие самопересечения.
    pts: (n,2) float. Решётка 2px для отбраковки пар сегментов."""
    n = len(pts)
    if n < 4:
        return False
    segs = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    bbox = [(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
            for a, b in segs]
    cell = 2.0
    grid = {}
    for i, (x0, y0, x1, y1) in enumerate(bbox):
        for gx in range(int(x0 // cell), int(x1 // cell) + 1):
            for gy in range(int(y0 // cell), int(y1 // cell) + 1):
                grid.setdefault((gx, gy), []).append(i)
    tested = set()
    for bucket in grid.values():
        if len(bucket) < 2:
            continue
        for k in range(len(bucket)):
            i = bucket[k]
            for m in range(k + 1, len(bucket)):
                j = bucket[m]
                if j <= i:
                    i, j = j, i
                if (i, j) in tested:
                    continue
                tested.add((i, j))
                if j == i + 1 or (i == 0 and j == n - 1):
                    continue
                a, b = segs[i]
                c, d = segs[j]
                if _proper_cross(a, b, c, d):
                    return True
    return False


def ratio_ok(cnt, ratio):
    a = max(abs(cv2.contourArea(cnt)), 1.0)
    p = cv2.arcLength(cnt, True)
    high = 1.0 + p / (2.0 * a) + 0.12
    return RATIO_LOW <= ratio <= high


def safe_loop(cnt):
    """Упрощённый контур, гарантированно простой (если возможно).
    Возвращает (approx, ratio|None): ratio=None — только полилиния, в HATCH
    петлю не класть."""
    best = None
    for eps in EPS_LADDER:
        ap = cv2.approxPolyDP(cnt, eps, True)
        if len(ap) < 3:
            continue
        r = loop_fill_ratio(ap)
        if ratio_ok(ap, r) and not has_self_intersection(
                ap[:, 0, :].astype(np.float64)):
            return ap, r
        best = (ap, r) if best is None or abs(1.0 - r) < abs(1.0 - best[1]) \
            else best
    # простую петлю получить не удалось — оставим точную (eps~0) как полилинию
    exact = cv2.approxPolyDP(cnt, 0.05, True)
    return (exact if len(exact) >= 3 else (best[0] if best else cnt)), None


def trace_components(mask):
    """-> список компонентов: [(outer_cnt, [hole_cnt, ...]), ...]"""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    if contours is None:
        return []
    hierarchy = hierarchy[0]
    roots = [i for i in range(len(contours)) if hierarchy[i][3] == -1]
    comps = []
    for ri in roots:
        outer = contours[ri]
        if cv2.contourArea(outer) < MIN_AREA or cv2.arcLength(outer, True) < MIN_PERIM:
            continue
        def emit(c):
            return safe_loop(c)
        loops = []
        lp = emit(outer)
        if lp:
            loops.append(lp)
        # все потомки (дырки и островки глубже) — как петли границы
        stack = [hierarchy[ri][2]]
        while stack:
            ci = stack.pop()
            if ci == -1:
                continue
            c = contours[ci]
            if cv2.contourArea(c) >= MIN_AREA and cv2.arcLength(c, True) >= MIN_PERIM:
                lp = emit(c)
                if lp:
                    loops.append(lp)
            stack.append(hierarchy[ci][0])  # next sibling
            stack.append(hierarchy[ci][2])  # first child
        if loops:
            comps.append(loops)
    return comps


def main(pdf_path, out_dir):
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    img, img_rect, page_rect = extract_page_image(pdf_path)
    W, H = img.size
    print(f"[1] растр: {W}x{H} px, размещён в {img_rect} pt, лист {page_rect}")

    pt_per_px_x = img_rect.width / W
    pt_per_px_y = img_rect.height / H
    MM_PER_PT = 25.4 / 72.0 * PRINT_SCALE

    def to_dxf(px, py):  # координаты в апскейл-пространстве -> мм (Y вверх)
        x_pt = img_rect.x0 + (px / UPSCALE) * pt_per_px_x
        y_pt = img_rect.y0 + (py / UPSCALE) * pt_per_px_y
        return (round(x_pt * MM_PER_PT, 2),
                round((page_rect.height - y_pt) * MM_PER_PT, 2))

    masks = segment_colors(np.array(img))
    print(f"[2] покрытие слоёв, px: { {k: int((v > 0).sum()) for k, v in masks.items()} }")

    doc = ezdxf.new(DXF_VERSION, setup=True)
    doc.units = 4
    doc.header["$INSUNITS"] = 4
    doc.header["$MEASUREMENT"] = 1
    msp = doc.modelspace()

    layers = {  # маска -> (слой, цвет ACI, заливка TrueColor, толщина 1/100 мм)
        "walls": ("A_WALLS", 1, FILL_WALLS, 35),
        "blue":  ("A_LITERALS_blue", 5, FILL_BLUE, 25),
        "dark":  ("A_DRAWING", 7, FILL_BLACK, 18),
    }
    for _m, (lname, aci, _tc, lw) in layers.items():
        if lname not in doc.layers:
            doc.layers.add(lname, color=aci, lineweight=lw)
    doc.layers.add("RASTER_UNDERLAY", color=8)
    doc.layers.add("NOTES", color=3)

    def big_fill_loops(labels_map, lab):
        """Заливка большого компонента: эродированная маска без защипов."""
        ys, xs = np.where(labels_map == lab)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        crop = ((labels_map[y0:y1 + 1, x0:x1 + 1] == lab) * 255).astype(np.uint8)
        er = cv2.erode(crop, np.ones((3, 3), np.uint8))
        contours, _h = cv2.findContours(er, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        loops = []
        for c in contours or []:
            if cv2.contourArea(c) < MIN_AREA or cv2.arcLength(c, True) < MIN_PERIM:
                continue
            ap, ratio = safe_loop(c)
            if len(ap) < 3 or ratio is None:
                continue
            loops.append(ap + np.array([[[x0, y0]]]))
        return loops

    total_h = total_pl = 0
    for mkey, (lname, aci, tc, _lw) in layers.items():
        mask = upscale_mask(masks[mkey])
        comps = trace_components(mask)
        _nlab, labels_map = cv2.connectedComponents(
            (mask > 0).astype(np.uint8), connectivity=8)
        n_h = n_pl = n_skip = 0
        for loops in comps:
            outer_raw = loops[0][0]
            comp_area = abs(cv2.contourArea(outer_raw))

            def verts(l):
                return [to_dxf(float(p[0][0]), float(p[0][1])) for p in l]

            if comp_area >= BIG_AREA:
                # большой компонент: заливка — по эродированной маске,
                # границы (полилинии) — по исходной, всё точно
                lab = int(labels_map[int(outer_raw[0][0][1]),
                                     int(outer_raw[0][0][0])])
                fill_loops = big_fill_loops(labels_map, lab)
                if fill_loops:
                    try:
                        hatch = msp.add_hatch(color=aci,
                                              dxfattribs={"layer": lname})
                        hatch.set_pattern_fill("SOLID")
                        hatch.rgb = tc
                        for lp in fill_loops:
                            hatch.paths.add_polyline_path(verts(lp),
                                                          is_closed=True)
                        n_h += 1
                    except Exception:
                        pass
                pl_loops = [l for l, _r in loops]
            else:
                # малый компонент: заливка по своим контурам с фильтром
                good = [(l, r) for l, r in loops]
                if good[0][1] is not None:
                    try:
                        hatch = msp.add_hatch(color=aci,
                                              dxfattribs={"layer": lname})
                        hatch.set_pattern_fill("SOLID")
                        hatch.rgb = tc
                        for l, r in good:
                            if r is not None:
                                hatch.paths.add_polyline_path(verts(l),
                                                              is_closed=True)
                            else:
                                n_skip += 1
                        n_h += 1
                    except Exception:
                        pass
                else:
                    n_skip += 1
                pl_loops = [l for l, _r in loops]
            # полилинии границ — всегда (геометрия для привязок)
            for lp in pl_loops:
                try:
                    msp.add_lwpolyline(verts(lp), close=True,
                                       dxfattribs={"layer": lname, "color": aci})
                    n_pl += 1
                except Exception:
                    pass
        total_h += n_h; total_pl += n_pl
        print(f"[3] {lname}: {len(comps)} компонентов -> {n_h} hatch, "
              f"{n_pl} полилиний (отфильтровано вырожденных: {n_skip})")

    # ---- растровая подложка ----
    png_name = base + " — подложка.png"
    img.save(os.path.join(out_dir, png_name), optimize=True)
    image_def = doc.add_image_def(png_name, size_in_pixel=(W, H))
    x0_mm = img_rect.x0 * MM_PER_PT
    y0_mm = (page_rect.height - img_rect.y1) * MM_PER_PT
    w_mm = img_rect.width * MM_PER_PT
    h_mm = img_rect.height * MM_PER_PT
    msp.add_image(image_def, insert=(x0_mm, y0_mm), size_in_units=(w_mm, h_mm),
                  dxfattribs={"layer": "RASTER_UNDERLAY"})
    print(f"[4] подложка {png_name}: ({x0_mm:.0f},{y0_mm:.0f}) {w_mm:.0f}x{h_mm:.0f} мм")

    msp.add_text(
        "Векторизация растрового PDF (М 1:200 на листе ANSI C 17x22 in). "
        "Единицы чертежа — мм реального здания. Слои: A_WALLS (стены), "
        "A_DRAWING (чертёж/текст кривыми), A_LITERALS_blue (синие отметки). "
        "Оригинал-растр на выключаемом слое RASTER_UNDERLAY.",
        dxfattribs={"layer": "NOTES", "height": 500},
    ).set_placement((x0_mm, y0_mm - 1500))

    out_dxf = os.path.join(out_dir, base + ".dxf")
    doc.saveas(out_dxf)
    print(f"[5] сохранено {out_dxf}: {total_h} hatch, {total_pl} полилиний")
    return out_dxf


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "План 1 этажа.pdf"
    main(src, os.path.dirname(os.path.abspath(src)))
