# -*- coding: utf-8 -*-
"""Рендер DXF (HATCH/LWPOLYLINE/LINE) в пиксели исходного растра (1:1).
Заливка штриховки — nested: глубина вложенности петли определяет заливку."""
import sys

import cv2
import numpy as np
import ezdxf

MM_PER_PT = 25.4 / 72.0 * 200.0
X0PT, Y0PT, PAGE_H = 41.76, 49.44, 1584.0
PT_PER_PX_X, PT_PER_PX_Y = 1182.24 / 2463.0, 1485.6 / 3095.0

FILL = {"A_WALLS": (241, 183, 183), "A_LITERALS_blue": (0, 0, 205),
        "A_DRAWING": (40, 40, 40)}
EDGE = {"A_WALLS": (200, 30, 30), "A_LITERALS_blue": (0, 0, 180),
        "A_DRAWING": (0, 0, 0)}


def to_px(pt):
    return (float((pt[0] / MM_PER_PT - X0PT) / PT_PER_PX_X),
            float((PAGE_H - pt[1] / MM_PER_PT - Y0PT) / PT_PER_PX_Y))


def render_hatch(canvas, e):
    loops = []
    for p in e.paths:
        if p.path_type_flags & 2:
            loops.append(np.array([to_px((v[0], v[1])) for v in p.vertices],
                                  np.float32))
    if not loops:
        return 0
    rects = [cv2.boundingRect(lp) for lp in loops]
    depth = []
    for i, lp in enumerate(loops):
        cx, cy = lp[0]
        xi, yi, wi, hi = rects[i]
        d = 0
        for j, other in enumerate(loops):
            if i == j:
                continue
            xj, yj, wj, hj = rects[j]
            inside_bbox = (xj <= cx <= xj + wj) and (yj <= cy <= yj + hj)
            strictly_smaller = (wj * hj) < (wi * hi)
            if inside_bbox and strictly_smaller:
                if cv2.pointPolygonTest(other, (cx + 0.01, cy + 0.01),
                                        False) > 0:
                    d += 1
        depth.append(d)
    # рисуем в обрезанном окне (скорость)
    x0 = max(0, min(r[0] for r in rects) - 2)
    y0 = max(0, min(r[1] for r in rects) - 2)
    x1 = min(canvas.shape[1], max(r[0] + r[2] for r in rects) + 2)
    y1 = min(canvas.shape[0], max(r[1] + r[3] for r in rects) + 2)
    tmp = np.zeros((y1 - y0, x1 - x0), np.uint8)
    off = np.array([[[x0, y0]]], np.int32)
    for i in sorted(range(len(loops)), key=lambda k: -depth[k]):
        cv2.fillPoly(tmp, [(loops[i] - [x0, y0]).astype(np.int32)],
                     255 if depth[i] % 2 == 0 else 0)
    canvas[y0:y1, x0:x1][tmp > 0] = FILL.get(e.dxf.layer, (0, 0, 0))
    return 1


def main(dxf_path, out_path):
    doc = ezdxf.readfile(dxf_path)
    H, W = 3095, 2463
    canvas = np.full((H, W, 3), 255, np.uint8)
    n_h = n_pl = n_li = 0
    for e in doc.modelspace():
        if e.dxftype() == "HATCH":
            n_h += render_hatch(canvas, e)
    for e in doc.modelspace():
        lay = e.dxf.layer
        if e.dxftype() == "LWPOLYLINE":
            pts = np.array([to_px(p) for p in e.get_points("xy")], np.int32)
            cv2.polylines(canvas, [pts], bool(e.closed) and len(pts) > 2,
                          EDGE.get(lay, (60, 60, 60)), 1)
            n_pl += 1
        elif e.dxftype() == "LINE":
            cv2.line(canvas,
                     tuple(map(int, to_px(e.dxf.start))),
                     tuple(map(int, to_px(e.dxf.end))),
                     EDGE.get(lay, (60, 60, 60)), 1)
            n_li += 1
    print(f"render: {n_h} hatch, {n_pl} pline, {n_li} line")
    cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print("saved", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
