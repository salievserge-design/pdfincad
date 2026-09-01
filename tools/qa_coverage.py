# -*- coding: utf-8 -*-
"""QA: насколько векторный DXF покрывает исходный растр (по каждому слою).

Метрики по маске слоя (в пикселях исходного растра):
  recall    = доля исходных пикселей, у которых в радиусе D есть векторный пиксель
  precision = доля векторных пикселей, у которых в радиусе D есть исходный пиксель
recall ~ "ничего не потерялось".
Дополнительно сохраняет карты потерь /tmp/qa_lost_<слой>.png
"""
import sys
import cv2
import numpy as np
import ezdxf
from PIL import Image

D = 2  # допуск, px (линия толщиной 1-2 px трассуется обводкой двух кромок)

# геопривязка растра (как в raster2dxf.py)
MM_PER_PT = 25.4 / 72.0 * 200.0
X0PT, Y0PT = 41.76, 49.44          # левый верх картинки на странице, pt (сверху)
PAGE_H = 1584.0
PT_PER_PX_X, PT_PER_PX_Y = 1182.24 / 2463.0, 1485.6 / 3095.0


def to_px(pt):
    x_pt = pt[0] / MM_PER_PT                      # pt от левого края страницы
    y_pt_from_top = PAGE_H - pt[1] / MM_PER_PT    # pt от верха страницы
    return (int(round((x_pt - X0PT) / PT_PER_PX_X)),
            int(round((y_pt_from_top - Y0PT) / PT_PER_PX_Y)))


def main(dxf_path, png_path):
    img = np.array(Image.open(png_path).convert("RGB"))
    H, W = img.shape[:2]
    r, g, b = (img[:, :, i].astype(np.int16) for i in range(3))
    v = np.maximum(np.maximum(r, g), b)
    red = ((r > g + 25) & (r > b + 25) & (r > 90))
    blue = ((b > r + 20) & (b > g + 20) & (b > 80))
    dark = ((v < 235) & ~red & ~blue)
    src = {"A_WALLS": red, "A_LITERALS_blue": blue, "A_DRAWING": dark}

    vec = {k: np.zeros((H, W), np.uint8) for k in src}
    doc = ezdxf.readfile(dxf_path)
    n_line = n_pl = 0
    for e in doc.modelspace():
        lay = e.dxf.layer
        if lay not in vec:
            continue
        if e.dxftype() == "LWPOLYLINE":
            pts = np.array([to_px(p) for p in e.get_points("xy")], np.int32)
            cv2.polylines(vec[lay], [pts], bool(e.closed) and len(pts) > 2, 255, 1)
            n_pl += 1
        elif e.dxftype() == "LINE":
            cv2.line(vec[lay], to_px(e.dxf.start), to_px(e.dxf.end), 255, 1)
            n_line += 1
    print(f"прочитано из DXF: {n_pl} LWPOLYLINE, {n_line} LINE")

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * D + 1, 2 * D + 1))
    for lay in src:
        s = src[lay].astype(np.uint8) * 255
        vc = vec[lay]
        s_dil = cv2.dilate(s, k)
        v_dil = cv2.dilate(vc, k)
        s_n = int((s > 0).sum()); v_n = int((vc > 0).sum())
        recall = ((s > 0) & (v_dil > 0)).sum() / max(s_n, 1)
        precision = ((vc > 0) & (s_dil > 0)).sum() / max(v_n, 1)
        print(f"{lay:18s} src_px={s_n:8d} vec_px={v_n:8d} "
              f"recall={recall * 100:6.2f}%  precision={precision * 100:6.2f}%")
        # карта потерь
        lost = ((s > 0) & ~(v_dil > 0)).astype(np.uint8) * 255
        canvas = (img * 0.35 + 160).astype(np.uint8)
        canvas[cv2.dilate(lost, np.ones((3, 3), np.uint8)) > 0] = (30, 30, 255)
        cv2.imwrite(f"/tmp/qa_lost_{lay}.png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
