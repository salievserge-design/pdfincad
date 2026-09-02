# -*- coding: utf-8 -*-
"""Точечная проверка заливки: выборка внутренних пикселей исходных масок
(>=3px от края компонента), проверка — попадает ли точка внутрь границ
HATCH соответствующего слоя в DXF. Это мерило 'зальёт ли AutoCAD'."""
import sys
import cv2
import numpy as np
import ezdxf
from PIL import Image

MM_PER_PT = 25.4 / 72.0 * 200.0
X0PT, Y0PT, PAGE_H = 41.76, 49.44, 1584.0
PT_PER_PX_X, PT_PER_PX_Y = 1182.24 / 2463.0, 1485.6 / 3095.0


def to_px(pt):
    x_pt = pt[0] / MM_PER_PT
    y_pt = PAGE_H - pt[1] / MM_PER_PT
    return ((x_pt - X0PT) / PT_PER_PX_X, (y_pt - Y0PT) / PT_PER_PX_Y)


def main(dxf_path, png_path, n_samples=4000):
    rng = np.random.default_rng(42)
    img = np.array(Image.open(png_path).convert("RGB"))
    H, W = img.shape[:2]
    r, g, b = (img[:, :, i].astype(np.int16) for i in range(3))
    v = np.maximum(np.maximum(r, g), b)
    masks = {
        "A_WALLS": ((r > g + 25) & (r > b + 25) & (r > 90)),
        "A_LITERALS_blue": ((b > r + 20) & (b > g + 20) & (b > 80)),
        "A_DRAWING": ((v < 235) & ~((r > g + 25) & (r > b + 25) & (r > 90))
                      & ~((b > r + 20) & (b > g + 20) & (b > 80))),
    }
    # границы (полилинии) тоже считаем заполненными допуском 1px —
    # в AutoCAD линия верх заливки видна в любом случае
    k = np.ones((3, 3), np.uint8)

    # собираем петли HATCH и полилиний по слоям
    loops = {k2: [] for k2 in masks}
    pline_img = {k2: np.zeros((H, W), np.uint8) for k2 in masks}
    for e in ezdxf.readfile(dxf_path).modelspace():
        lay = e.dxf.layer
        if lay not in loops:
            continue
        if e.dxftype() == "HATCH":
            for p in e.paths:
                if p.path_type_flags & 2:
                    loops[lay].append(
                        np.array([to_px((vv[0], vv[1])) for vv in p.vertices],
                                 np.float32))
        elif e.dxftype() == "LWPOLYLINE":
            pts = np.array([to_px(pp) for pp in e.get_points("xy")], np.int32)
            cv2.polylines(pline_img[lay], [pts], bool(e.closed), 255, 1)

    for lay, msk in masks.items():
        # внутренние пиксели: эрозия маски на 3px
        inner = cv2.erode(msk.astype(np.uint8) * 255, np.ones((7, 7), np.uint8))
        ys, xs = np.where(inner > 0)
        total = len(xs)
        if total == 0:
            print(f"{lay}: нет внутренних пикселей")
            continue
        idx = rng.choice(total, size=min(n_samples, total), replace=False)
        inside_h = near_pl = 0
        lps = loops[lay]
        pl_dil = cv2.dilate(pline_img[lay], k)
        for i in idx:
            x, y = float(xs[i]), float(ys[i])
            depth = sum(1 for lp in lps
                        if cv2.pointPolygonTest(lp, (x, y), False) >= 0)
            if depth % 2 == 1:      # nested: нечётная вложенность = залито
                inside_h += 1
            elif pl_dil[int(y), int(x)]:
                near_pl += 1
        print(f"{lay:18s} внутр.пикселей={total:7d}; залито hatch: "
              f"{100*inside_h/len(idx):6.2f}%; покрыто линией (+1px): "
              f"{100*near_pl/len(idx):6.2f}%; ничем: "
              f"{100*(len(idx)-inside_h-near_pl)/len(idx):6.2f}%")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
