# -*- coding: utf-8 -*-
"""
OCR скана -> редактируемые TEXT-объекты в DXF.
- 4 прогона Tesseract (повороты 0/90/180/270) -> находим и вертикальный текст
- строки из TSV-групп (block/par/line), дедупликация по IoU боксов (лучший conf)
- DXF: слой A-TEXT-OCR (TEXT, выравнивание FIT), контуры шрифта со слоя
  A-PLAN-GRAPHICS переезжают на A-TEXT-OUTLINE (можно заморозить — ничего не теряется)
"""
import subprocess, io, math, sys, os
import numpy as np
from PIL import Image
import ezdxf
from ezdxf.enums import TextEntityAlignment

TESS = "/home/user/work_dl/tess-install/bin/tesseract"
IMG = "work/plan_raw.png"
DXF = "План 1 этажа.dxf"
UP = 2                 # апскейл картинки для OCR
CONF_MIN = 30          # фильтр уверенности
H_MIN, H_MAX = 5, 60   # высота строки в px оригинала (лишнее отсекаем)
TESSDATA = "/home/user/work_dl/tess-best/share/tessdata"

# --- система координат (как в vectorize.py) ---
origin_x_mm = 41.76 * (25.4 / 72) * 200
origin_top_mm = 49.44 * (25.4 / 72) * 200
MPP = (1182.24 / 2463) * (25.4 / 72) * 200     # мм/px в исходном растре

im0 = Image.open(IMG).convert("L")
W, H = im0.size
big = im0.resize((W * UP, H * UP), Image.LANCZOS)

VAR_VARIANTS = ("gray", "bin")

def tsv_for_rotation(k, variant):
    """k*90 CCW -> TSV-таблица (строки-списки)."""
    rim = big.rotate(k * 90, expand=True) if k else big
    a = np.asarray(rim)
    if variant == "bin":                      # ч/б: серые стены не отвлекают OCR
        a = np.where(a < 150, 0, 255).astype(np.uint8)
    wp, hp = rim.size
    fn = f"/tmp/ocr_{variant}_r{k}.pgm"
    with open(fn, "wb") as f:   # сырой PGM (P5) — leptonica читает нативно
        f.write(f"P5\n{wp} {hp}\n255\n".encode())
        f.write(a.tobytes())
    env = dict(os.environ, TESSDATA_PREFIX=TESSDATA)
    out = subprocess.run(
        [TESS, fn, "stdout", "-l", "rus", "--psm", "11", "/home/user/work_dl/tess-install/share/tessdata/configs/tsv"],
        capture_output=True, text=True, env=env).stdout
    rows = []
    import csv
    for r in csv.DictReader(io.StringIO(out), delimiter="\t"):
        rows.append(r)
    return rows, (wp, hp)

def inv_map(k, xr, yr, wp, hp):
    """точка из повёрнутой картинки -> координаты исходного растра (px, y вниз)."""
    if k == 0: return xr, yr
    if k == 1: return hp - 1 - yr, xr          # было rotate(90 CCW)
    if k == 2: return wp - 1 - xr, hp - 1 - yr
    if k == 3: return yr, wp - 1 - xr          # rotate(270 CCW)
    raise AssertionError

lines = []
for k in (0, 1, 2, 3):
    for variant in VAR_VARIANTS:
        rows, (wp, hp) = tsv_for_rotation(k, variant)
        groups = {}
        for r in rows:
            if r.get("level") != "5":
                continue
            if None in r or r.get(None):   # битая строка TSV (табуляции внутри text)
                continue
            txt = (r["text"] or "").strip()
            try:
                conf = float(r["conf"])
            except (ValueError, TypeError):
                continue
            if not txt or conf < CONF_MIN:
                continue
            if not any(ch.isalnum() for ch in txt):
                continue
            key = (r["block_num"], r["par_num"], r["line_num"], variant)
            groups.setdefault(key, []).append(r)
        for key, ws in groups.items():
            ws.sort(key=lambda r: int(r["left"]))
            text = " ".join(w["text"].strip() for w in ws).strip()
            confs = [float(w["conf"]) for w in ws]
            conf = sum(confs) / len(confs)
            x0 = min(int(w["left"]) for w in ws) / UP
            y0 = min(int(w["top"]) for w in ws) / UP
            x1 = max(int(w["left"]) + int(w["width"]) for w in ws) / UP
            y1 = max(int(w["top"]) + int(w["height"]) for w in ws) / UP
            hpx = y1 - y0
            if hpx < H_MIN or hpx > H_MAX:
                continue
            # фильтры мусора и ложных срабатываний
            if "|" in text or len(text) > 48 or text.count(" ") > 4:
                continue
            if len(text) > 10 and sum(ch2.isdigit() for ch2 in text) > len(text) * 0.8 \
                    and " " in text:
                continue  # "числительная каша" — обрывки TSV
            has_digit = any(ch.isdigit() for ch in text)
            letters = sum(ch.isalpha() for ch in text)
            if has_digit:
                if conf < CONF_MIN:                     # числа — главное содержание плана
                    continue
            else:
                # чисто буквенные строки: только уверенные осмысленные подписи
                if conf < 72 or letters < 4:
                    continue
            if len(text) == 1 and conf < 55:
                continue
            wpx = x1 - x0
            if wpx / max(1, len(text)) > hpx * 1.25:    # «размазанная» строка
                continue
            # baseline start/end и полный бокс проецируем через инверсию поворота
            ax, ay = inv_map(k, x0 * UP, y1 * UP, wp, hp)   # левый низ бокса
            bx, by = inv_map(k, x1 * UP, y1 * UP, wp, hp)   # правый низ бокса
            cx, cy = inv_map(k, x1 * UP, y0 * UP, wp, hp)   # правый верх
            dx, dy = inv_map(k, x0 * UP, y0 * UP, wp, hp)   # левый верх
            ang = math.degrees(math.atan2(-(by - ay), bx - ax))  # y вниз -> y вверх
            xs = (ax, bx, cx, dx); ys = (ay, by, cy, dy)
            lines.append({"text": text, "conf": conf,
                          "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                          "ax": ax, "ay": ay, "bx": bx, "by": by,
                          "rot": ang % 360,
                          "box": (min(xs), min(ys), max(xs), max(ys))})
print("строк OCR (сырых):", len(lines))

# --- дедупликация по пересечению боксов (в исходных координатах) ---
def iou(b1, b2):
    x0 = max(b1[0], b2[0]); y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2]); y1 = min(b1[3], b2[3])
    if x1 <= x0 or y1 <= y0: return 0.0
    inter = (x1 - x0) * (y1 - y0)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1]); a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / min(a1, a2)

keep = []
for l in sorted(lines, key=lambda t: -t["conf"]):
    if any(iou(l["box"], o["box"]) > 0.45 for o in keep):
        continue
    keep.append(l)
print("после дедупликации:", len(keep))

def sanitize(txt):
    """DXF R2000/cp1251-совместимая строка: без управляющих и не-код.page символов."""
    out = []
    for ch in txt:
        if ch in "\r\n\t" or ord(ch) < 32:
            out.append(" ")
            continue
        try:
            ch.encode("cp1251")
            out.append(ch)
        except UnicodeEncodeError:
            out.append(" ")   # недопустимый для cp1251 символ
    s = " ".join("".join(out).split())
    return s.strip()

# --- в DXF ---
doc = ezdxf.readfile(DXF)
msp = doc.modelspace()
if not doc.layers.has_entry("A-TEXT-OCR"):
    doc.layers.add("A-TEXT-OCR", color=3)      # зелёный: свежераспознанный текст
# очистка от прошлого прогона
for e in list(msp.query("TEXT[layer=='A-TEXT-OCR']")):
    msp.delete_entity(e)

def to_mm(px, py):
    return origin_x_mm + px * MPP, origin_top_mm + (H - py) * MPP

# --- калибровка baseline/высоты по контурам глифов внутри OCR-бокса ---
# предрасчёт боксов контуров слоя A-PLAN-GRAPHICS в px исходного растра
conts = []
for e in msp.query("LWPOLYLINE[layer=='A-PLAN-GRAPHICS']"):
    pts = np.array(list(e.get_points("xy")), dtype=float)
    if len(pts) == 0:
        continue
    xs = (pts[:, 0] - origin_x_mm) / MPP
    ys = H - (pts[:, 1] - origin_top_mm) / MPP
    conts.append((xs, ys))
print("контуров для калибровки:", len(conts))

def refine(l):
    """посадить baseline/высоту на фактические контуры знаков; fallback — бокс OCR.
    Работаем в CAD-координатах (мм, y вверх)."""
    x0, y0, x1, y1 = l["x0"] - 2, l["y0"] - 2, l["x1"] + 2, l["y1"] + 2
    sel = []
    for xs, ys in conts:
        if xs.min() >= x0 and xs.max() <= x1 and ys.min() >= y0 and ys.max() <= y1:
            sel.append((xs, ys))
    r = math.radians(l["rot"])
    c, s = math.cos(r), math.sin(r)
    if not sel:
        h_mm = ((l["y1"] - l["y0"]) * MPP) * 0.92
        pa = to_mm(l["ax"], l["ay"]); pb = to_mm(l["bx"], l["by"])
        return pa, pb, l["rot"], h_mm
    X = np.concatenate([t[0] for t in sel]); Yscr = np.concatenate([t[1] for t in sel])
    Xmm = origin_x_mm + X * MPP
    Ymm = origin_top_mm + (H - Yscr) * MPP          # CAD y-вверх
    U = Xmm * c + Ymm * s                            # вдоль строки
    V = -Xmm * s + Ymm * c                           # поперёк (вверх от baseline)
    umin, umax = U.min(), U.max()
    vmin, vmax = V.min(), V.max()
    h_mm = max(vmax - vmin, (l["y1"] - l["y0"]) * MPP) * 1.02
    v0 = vmin + 0.14 * h_mm                          # baseline чуть выше низа чернил
    # обратный поворот (u,v) -> (X,Y): X = u*c - v*s ; Y = u*s + v*c
    pa = (umin * c - v0 * s, umin * s + v0 * c)
    pb = (umax * c - v0 * s, umax * s + v0 * c)
    return pa, pb, l["rot"], h_mm

n_txt = 0
boxes = []
for l in keep:
    pa, pb, rot, h_mm = refine(l)
    safe = sanitize(l["text"])
    if not safe:
        continue
    t = msp.add_text(safe, dxfattribs={
        "layer": "A-TEXT-OCR", "height": h_mm,
        "style": "Standard"})
    t.dxf.rotation = rot
    t.set_placement((pa[0], pa[1]), (pb[0], pb[1]),
                    align=TextEntityAlignment.FIT)
    n_txt += 1
    boxes.append((l["x0"] - 1, l["y0"] - 1, l["x1"] + 1, l["y1"] + 1))
print("TEXT добавлено:", n_txt)

# NB: контуры шрифта НЕ трогаем — они остаются на A-PLAN-GRAPHICS как
# гарантия полноты переноса. TEXT — это редактируемая копия поверх.

doc.saveas(DXF, encoding="cp1251")
print("DXF пересохранён")
