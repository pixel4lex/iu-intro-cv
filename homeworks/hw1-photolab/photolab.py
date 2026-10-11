# -*- coding: utf-8 -*-
"""ДЗ 1 «Фотолаборатория» — обработка цвета, хромакей и геометрия.

Запуск:  python photolab.py <команда> [аргументы]      (список команд — README.md)

За основу взят template/photolab.py из задания. Имена и сигнатуры
одиннадцати реализуемых функций сохранены. Самопроверка: ``pytest tests -q``.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

__version__ = "1.0"

# --------------------------------------------------------------- ввод-вывод ---


def imread(path):
    """Читает BGR uint8 и падает с внятной ошибкой (cv2.imread молча даёт None; кириллица в пути)."""
    if not os.path.exists(path):
        raise FileNotFoundError("файл не найден: %s" % path)
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("не удалось декодировать изображение: %s" % path)
    return img


def imwrite(path, img):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
    if not ok:
        raise ValueError("не удалось закодировать %s" % path)
    buf.tofile(path)
    return path


def as_uint8(x):
    """float → uint8 только через обрезку (правило курса: uint8 переполняется молча)."""
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


# ------------------------------------------------------------ 1. цветокоррекция ---


def gamma_correct(img, gamma):
    """Гамма-коррекция: out = 255 · (in / 255) ** gamma, через таблицу на 256 значений.

    gamma < 1 осветляет тени, gamma > 1 затемняет. Таблицу считаем сами (NumPy),
    применяем индексацией — cv2.LUT допустим, но не обязателен.
    """
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma должна быть положительным конечным числом")
    levels = np.arange(256, dtype=np.float32) / 255.0
    lut = as_uint8(255.0 * levels ** gamma)
    return lut[img]


def white_balance(img, method="grey-world", percentile=99.0):
    """Баланс белого без готовых функций: возвращает (кадр, коэффициенты усиления B, G, R).

    grey-world: средний цвет сцены серый → усиление канала = среднее всех средних / среднее канала.
    white-patch: самое яркое (percentile-й процентиль канала) — белое → усиление = 255 / процентиль.
    """
    x = img.astype(np.float32)
    if method == "grey-world":
        reference = x.mean(axis=(0, 1))
        target = reference.mean()
    elif method == "white-patch":
        if not 0 < percentile <= 100:
            raise ValueError("percentile должен быть в (0, 100]")
        reference = np.percentile(x, percentile, axis=(0, 1)).astype(np.float32)
        target = np.float32(255.0)
    else:
        raise ValueError("неизвестный метод баланса белого: %s" % method)
    # Отсутствующий канал не восстановить усилением; не делим на ноль.
    gains = np.ones(3, dtype=np.float32)
    np.divide(target, reference, out=gains, where=reference > 0)
    return as_uint8(x * gains), gains


def equalize_lab(img):
    """Глобальная эквализация яркости: канал L пространства Lab через cv2.equalizeHist (готовая функция)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.equalizeHist(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def clahe_lab(img, clip=2.0, tiles=8):
    """CLAHE по каналу L пространства Lab — цвет не трогаем, только яркость."""
    if not np.isfinite(clip) or clip <= 0 or tiles < 1:
        raise ValueError("clip и tiles должны быть положительными")
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def channel_hist(img):
    """Гистограммы каналов без cv2.calcHist: массив (256, C) — сколько пикселей каждого значения."""
    if img.ndim == 2:
        img = img[..., None]
    return np.stack([np.bincount(img[..., c].ravel(), minlength=256) for c in range(img.shape[2])], axis=1)


def tone_stats(img):
    """Числа для отчёта: среднее и процентили яркости (серого) — до и после коррекции."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    p = np.percentile(g, [1, 50, 99])
    return {"mean": round(float(g.mean()), 1), "p1": int(p[0]), "median": int(p[1]), "p99": int(p[2]),
            "clipped_black_pct": round(float(100 * (g == 0).mean()), 2), "clipped_white_pct": round(float(100 * (g == 255).mean()), 2)}


def local_contrast_report(img, out_eq, out_clahe):
    """Почему CLAHE, а не equalizeHist — измерением, как на слайде 24 L3.

    Тени — пиксели, чья исходная яркость ниже 20-го процентиля; света — выше 80-го.
    Возвращает σ в тенях и среднее в светах для обоих методов.
    """
    L0 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
    lo, hi = np.percentile(L0, [20, 80])
    shadows, lights = L0 <= lo, L0 >= hi
    rep = {}
    for name, im in (("исходный", img), ("equalizeHist", out_eq), ("CLAHE", out_clahe)):
        L = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0]
        rep[name] = {"shadow_std": round(float(L[shadows].std()), 1), "shadow_mean": round(float(L[shadows].mean()), 1),
                     "light_mean": round(float(L[lights].mean()), 1), "light_std": round(float(L[lights].std()), 1)}
    return rep


def save_hist_figure(pairs, path, title=""):
    """Рисунок «до / после»: по строке на каждую пару (подпись, кадр), кривые B, G, R."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(pairs), 2, figsize=(9, 2.6 * len(pairs)), squeeze=False)
    for row, (name, im) in enumerate(pairs):
        ax = axes[row][0]; ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax.set_title(name, fontsize=9); ax.axis("off")
        h = channel_hist(im); ax = axes[row][1]
        for c, col in enumerate(("#0070C0", "#548235", "#C00000")):
            ax.plot(h[:, c], color=col, lw=1)
        ax.set_xlim(0, 255); ax.set_yticks([]); ax.set_title("гистограмма B, G, R", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- 2. хромакей ---


def hsv_box_from(bgr_pixels, margin=10, p=(1, 99)):
    """Границы HSV по пикселям эталона: H и S — процентили ± margin, V — открыт (≥ 40): яркость меняет свет."""
    if bgr_pixels.size == 0:
        raise ValueError("пустой прямоугольник образца")
    hsv = cv2.cvtColor(bgr_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lo = np.clip(np.floor(np.percentile(hsv, p[0], axis=0) - margin), 0, 255)
    hi = np.clip(np.ceil(np.percentile(hsv, p[1], axis=0) + margin), 0, 255)
    # Разворачиваем круг H после наибольшего незанятого промежутка.
    # Красные оттенки 179 и 1 должны дать короткий интервал через ноль.
    hues = np.sort(np.unique(hsv[:, 0].astype(np.float32)))
    gaps = np.diff(np.r_[hues, hues[0] + 180])
    start = hues[(int(np.argmax(gaps)) + 1) % len(hues)]
    unwrapped = (hsv[:, 0].astype(np.float32) - start) % 180 + start
    hue_lo = int(np.floor(np.percentile(unwrapped, p[0]) - margin))
    hue_hi = int(np.ceil(np.percentile(unwrapped, p[1]) + margin))
    if hue_hi - hue_lo >= 179:
        lo[0], hi[0] = 0, 179
    else:
        lo[0], hi[0] = hue_lo % 180, hue_hi % 180
    return (int(lo[0]), int(lo[1]), 40), (min(int(hi[0]), 179), int(hi[1]), 255)


def color_mask(bgr, lo, hi, open_k=5, close_k=9):
    """HSV-маска по границам (lo, hi) + открытие/закрытие; lo[0] > hi[0] — красный через 179 (занятие 3)."""
    if len(lo) != 3 or len(hi) != 3:
        raise ValueError("границы должны содержать H,S,V")
    if not (0 <= lo[0] <= 179 and 0 <= hi[0] <= 179
            and 0 <= lo[1] <= hi[1] <= 255 and 0 <= lo[2] <= hi[2] <= 255):
        raise ValueError("недопустимые границы HSV")
    if open_k < 0 or close_k < 0:
        raise ValueError("размеры ядер не могут быть отрицательными")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if lo[0] <= hi[0]:
        mask = cv2.inRange(hsv, tuple(lo), tuple(hi))
    else:
        upper = cv2.inRange(hsv, tuple(lo), (179, hi[1], hi[2]))
        lower = cv2.inRange(hsv, (0, lo[1], lo[2]), tuple(hi))
        mask = cv2.bitwise_or(upper, lower)
    for op, k in ((cv2.MORPH_OPEN, open_k), (cv2.MORPH_CLOSE, close_k)):
        if k > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.morphologyEx(mask, op, kernel)
    return mask


def largest_component(mask, min_area=200):
    """Маска крупнейшей компоненты (uint8 0/255) и её статистика; None — если компонент нет или она мала."""
    binary = (mask > 0).astype(np.uint8)
    count, labels, regions, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return None
    label = 1 + int(np.argmax(regions[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = (int(v) for v in regions[label])
    if area < min_area:
        return None
    component = np.where(labels == label, 255, 0).astype(np.uint8)
    return component, {"bbox": [x, y, w, h], "area": area, "components": count - 1,
                       "centroid": [float(v) for v in centroids[label]]}


def chroma_key(img, bg, lo, hi, open_k=5, close_k=9, feather=2.0):
    """Объект по цвету → на новый фон. Возвращает (композит, маска объекта, статистика)."""
    h, w = img.shape[:2]
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA if bg.shape[0] > h else cv2.INTER_LINEAR)
    if not np.isfinite(feather) or feather < 0:
        raise ValueError("feather должен быть конечным и неотрицательным")
    raw = color_mask(img, lo, hi, open_k=0, close_k=0)
    cleaned = color_mask(img, lo, hi, open_k, close_k)
    found = largest_component(cleaned)
    if found is None:
        raise ValueError("объект не найден или меньше 200 пикселей; проверьте границы HSV")
    mask, stats = found
    stats["raw_components"] = cv2.connectedComponents(raw, connectivity=8)[0] - 1
    stats["morph_components"] = stats["components"]
    # Заполняем только замкнутые дыры, а не фон, соприкасающийся с краем кадра.
    padded = np.pad(mask, 1, mode="constant")
    flooded = padded.copy()
    cv2.floodFill(flooded, None, (0, 0), 255)
    holes = cv2.bitwise_not(flooded)[1:-1, 1:-1]
    stats["holes_before_fill"] = cv2.connectedComponents(holes, connectivity=8)[0] - 1
    mask = cv2.bitwise_or(mask, holes)
    stats["holes"] = 0
    stats["area"] = int(np.count_nonzero(mask))
    stats["mask_pct"] = round(100.0 * stats["area"] / (w * h), 2)
    stats["components"] = 1
    alpha = mask.astype(np.float32) / 255.0
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=feather)
    alpha = alpha[..., None]
    composite = as_uint8(img.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha))
    return composite, mask, stats


# ---------------------------------------------------------------- 3. геометрия ---


def rotate_keep_all(img, deg, border=(255, 255, 255)):
    """Поворот на deg градусов вокруг центра с холстом по повёрнутым углам — ни один пиксель не отрезан."""
    h, w = img.shape[:2]
    if not np.isfinite(deg):
        raise ValueError("угол должен быть конечным")
    M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), deg % 360, 1.0)
    # Углы границ пикселей, а не только центры крайних пикселей.
    corners = np.float32([[-0.5, -0.5], [w - 0.5, -0.5],
                          [w - 0.5, h - 0.5], [-0.5, h - 0.5]])
    rotated = corners @ M[:, :2].T + M[:, 2]
    extent = np.ptp(rotated, axis=0)
    nw, nh = np.ceil(extent - 1e-9).astype(int)
    M[:, 2] += ((nw - w) / 2.0, (nh - h) / 2.0)
    return cv2.warpAffine(img, M, (int(nw), int(nh)), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def resize_aa(img, scale, method="area"):
    """Масштабирование с правильным антиалиасингом.

    Уменьшение: ``area`` — cv2.resize(INTER_AREA); ``blur`` — GaussianBlur σ ≈ 0.4 / scale, затем INTER_LINEAR;
    ``naive`` — INTER_LINEAR без размытия (так делать нельзя: алиасинг, это контроль для измерения).
    Увеличение — всегда INTER_CUBIC: там алиасинга нет, а кубическая даёт +2 дБ на фото (L4).
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale должен быть положительным конечным числом")
    if method not in ("area", "blur", "naive"):
        raise ValueError("неизвестный метод масштабирования: %s" % method)
    h, w = img.shape[:2]
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    if scale >= 1:
        return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    if method == "area":
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if method == "blur":
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=0.4 / scale)
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def lanczos_reference(img, scale):
    """Независимый эталон уменьшения — Lanczos из Pillow (как в демо 2.2 лекции 1 и 2.4 лекции 4)."""
    from PIL import Image
    h, w = img.shape[:2]
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return np.asarray(Image.fromarray(img[..., ::-1]).resize(size, Image.LANCZOS))[..., ::-1].copy()


def psnr(a, b):
    a = a.astype(np.float32) / 255.0; b = b.astype(np.float32) / 255.0
    e = float(np.mean((a - b) ** 2))
    return float("inf") if e == 0 else float(10 * np.log10(1.0 / e))


def order_corners(pts):
    """Четыре точки в любом порядке → float32 (4, 2) в порядке tl, tr, br, bl (занятие 4)."""
    points = np.asarray(pts, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("нужны четыре конечные точки формы (4, 2)")
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    # Циклическая сортировка не повторяет угол при совпадении сумм/разностей.
    start = int(np.lexsort((ordered[:, 0], ordered[:, 1], ordered.sum(axis=1)))[0])
    ordered = np.roll(ordered, -start, axis=0)
    edges = np.roll(ordered, -1, axis=0) - ordered
    following = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if not np.all(cross > 0):
        raise ValueError("углы должны образовывать выпуклый невырожденный четырёхугольник")
    return ordered


def rectify(img, corners, size=None, aspect=None):
    """Четыре угла → фронтальный вид и H (занятие 4): размер из size / aspect (W / H) / длин сторон в кадре."""
    src = order_corners(corners)
    if aspect is not None and (not np.isfinite(aspect) or aspect <= 0):
        raise ValueError("aspect должен быть положительным конечным числом")
    if size is None:
        width = max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3]))
        width = max(2, int(round(float(width))))
        height = width / aspect if aspect is not None else max(
            np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1]))
        size = (width, max(2, int(round(float(height)))))
    if len(size) != 2 or any(int(v) != v or v < 2 for v in size):
        raise ValueError("size должен содержать два целых размера не меньше 2")
    w, h = (int(v) for v in size)
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    H = cv2.getPerspectiveTransform(src, dst)
    front = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)
    return front, H


def pick_corners(img, title="4 klika po uglam, Enter - gotovo, r - zanovo"):
    """Окно OpenCV: четыре клика по углам → float32 (4, 2) в координатах кадра (только локально)."""
    pts, scale = [], min(1.0, 1200.0 / img.shape[1])

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x / scale, y / scale))

    cv2.namedWindow(title); cv2.setMouseCallback(title, on_click)
    while True:
        view = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        for p in pts:
            cv2.circle(view, (int(p[0] * scale), int(p[1] * scale)), 6, (0, 200, 255), -1)
        cv2.imshow(title, view)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("r"):
            pts.clear()
        if key in (13, 10, ord("q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
    if len(pts) != 4:
        raise ValueError("нужно ровно 4 клика, получено %d" % len(pts))
    return np.float32(pts)


# ----------------------------------------------------------- 4. контактный лист ---


def contact_sheet(images, captions, cols=None, cell=(320, 240), pad=8, caption_h=26):
    """Мозаика N изображений с подписями: каждое вписано в ячейку cell с сохранением пропорций."""
    n = len(images)
    if n == 0 or len(captions) != n:
        raise ValueError("нужны изображения и одна подпись на каждое")
    if cols is not None and (cols < 1 or int(cols) != cols):
        raise ValueError("cols должен быть положительным целым")
    if min(cell) < 1 or pad < 0 or caption_h < 1:
        raise ValueError("недопустимые размеры ячеек или отступов")
    cols = cols or int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    cw, ch = cell
    sheet = np.full((rows * (ch + caption_h + pad) + pad, cols * (cw + pad) + pad, 3), 255, np.uint8)
    for i, (img, caption) in enumerate(zip(images, captions)):
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            alpha = img[..., 3:4].astype(np.float32) / 255.0
            img = as_uint8(img[..., :3].astype(np.float32) * alpha + 255.0 * (1 - alpha))
        h, w = img.shape[:2]
        scale = min(cw / w, ch / h)
        size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        thumb = cv2.resize(img, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        x = pad + (i % cols) * (cw + pad)
        y = pad + (i // cols) * (ch + caption_h + pad)
        tx, ty = x + (cw - size[0]) // 2, y + (ch - size[1]) // 2
        sheet[ty:ty + size[1], tx:tx + size[0]] = thumb
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_width = cv2.getTextSize(str(caption), font, 0.45, 1)[0][0]
        font_scale = 0.45 * min(1.0, (cw - 4) / max(text_width, 1))
        cv2.putText(sheet, str(caption), (x + 2, y + ch + caption_h - 7),
                    font, font_scale, (30, 30, 30), 1, cv2.LINE_AA)
    return sheet


# --------------------------------------------------------------------- CLI ---


def _ints(s):
    return tuple(int(v) for v in s.split(","))


def _log(**kw):
    print(json.dumps(kw, ensure_ascii=False))


def cmd_gamma(a):
    img = imread(a.image); out = gamma_correct(img, a.gamma)
    imwrite(a.out, out)
    if a.hist:
        save_hist_figure([("до", img), ("после, γ = %g" % a.gamma, out)], a.hist)
    _log(command="gamma", gamma=a.gamma, before=tone_stats(img), after=tone_stats(out), out=a.out)


def cmd_wb(a):
    img = imread(a.image); out, gains = white_balance(img, a.method, a.percentile)
    imwrite(a.out, out)
    if a.hist:
        save_hist_figure([("до", img), ("после, %s" % a.method, out)], a.hist)
    means = lambda im: [round(float(v), 1) for v in im.reshape(-1, 3).mean(0)]   # noqa: E731
    _log(command="wb", method=a.method, gains_bgr=[round(float(g), 3) for g in gains], means_before_bgr=means(img), means_after_bgr=means(out), out=a.out)


def cmd_clahe(a):
    img = imread(a.image); out = clahe_lab(img, a.clip, a.tiles); eq = equalize_lab(img)
    imwrite(a.out, out)
    if a.compare:
        imwrite(a.compare, np.hstack([img, eq, out]))
    if a.hist:
        save_hist_figure([("до", img), ("equalizeHist", eq), ("CLAHE clip %g, tiles %d" % (a.clip, a.tiles), out)], a.hist)
    _log(command="clahe", clip=a.clip, tiles=a.tiles, local_contrast=local_contrast_report(img, eq, out), out=a.out)


def cmd_chromakey(a):
    img = imread(a.image); bg = imread(a.bg)
    if a.sample:
        if len(a.sample) != 4:
            raise ValueError("sample должен содержать x,y,w,h")
        x, y, w, h = a.sample
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > img.shape[1] or y + h > img.shape[0]:
            raise ValueError("прямоугольник образца должен целиком лежать внутри кадра")
        lo, hi = hsv_box_from(img[y:y + h, x:x + w].reshape(-1, 3))
    else:
        lo, hi = a.lo, a.hi
    comp, mask, stats = chroma_key(img, bg, lo, hi, a.open, a.close, a.feather)
    imwrite(a.out, comp)
    if a.mask:
        imwrite(a.mask, mask)
    _log(command="chromakey", lo=list(lo), hi=list(hi), **stats, out=a.out)


def cmd_rotate(a):
    img = imread(a.image); out = rotate_keep_all(img, a.deg)
    imwrite(a.out, out)
    h, w = img.shape[:2]; nh, nw = out.shape[:2]
    _log(command="rotate", deg=a.deg, size_in=[w, h], size_out=[nw, nh], canvas_empty_pct=round(100 * (1 - w * h / (nw * nh)), 1), out=a.out)


def cmd_resize(a):
    img = imread(a.image); out = resize_aa(img, a.scale, a.method)
    imwrite(a.out, out)
    rep = {"command": "resize", "scale": a.scale, "method": a.method, "size_out": [out.shape[1], out.shape[0]], "out": a.out}
    if a.report and a.scale < 1:
        ref = lanczos_reference(img, a.scale)
        rep["psnr_vs_lanczos_db"] = {m: round(psnr(ref, resize_aa(img, a.scale, m)), 1) for m in ("naive", "blur", "area")}
    _log(**rep)


def cmd_rectify(a):
    img = imread(a.image)
    corners = pick_corners(img) if a.clicks else np.float32(a.corners).reshape(4, 2)
    size = tuple(int(v) for v in a.size.lower().split("x")) if a.size else None
    front, H = rectify(img, corners, size, a.aspect)
    imwrite(a.out, front)
    c = order_corners(corners)
    side = lambda p, q: round(float(np.linalg.norm(p - q)), 1)   # noqa: E731
    _log(command="rectify", corners_ordered=c.tolist(), sides_px={"top": side(c[0], c[1]), "right": side(c[1], c[2]), "bottom": side(c[3], c[2]), "left": side(c[0], c[3])},
         size_out=[front.shape[1], front.shape[0]], aspect_out=round(front.shape[1] / front.shape[0], 3), out=a.out)


def cmd_sheet(a):
    # PowerShell не разворачивает маски файлов для внешних программ.
    paths = []
    for pattern in a.images:
        matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        if not matches:
            raise ValueError("нет файлов по шаблону: %s" % pattern)
        paths.extend(matches)
    images = [imread(p) for p in paths]
    caps = ["%s %dx%d" % (os.path.basename(p), im.shape[1], im.shape[0]) for p, im in zip(paths, images)]
    sheet = contact_sheet(images, caps, a.cols, (a.cell_w, a.cell_h))
    imwrite(a.out, sheet)
    _log(command="sheet", n=len(images), cols=a.cols or int(np.ceil(np.sqrt(len(images)))), size_out=[sheet.shape[1], sheet.shape[0]], out=a.out)


def build_parser():
    p = argparse.ArgumentParser(prog="photolab", description="ДЗ 1 «Фотолаборатория»: цвет, хромакей, геометрия, контактный лист")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("gamma", help="гамма-коррекция"); s.add_argument("image"); s.add_argument("--gamma", type=float, required=True)
    s.add_argument("-o", "--out", required=True); s.add_argument("--hist", help="PNG с гистограммами до/после"); s.set_defaults(fn=cmd_gamma)

    s = sub.add_parser("wb", help="баланс белого"); s.add_argument("image"); s.add_argument("--method", choices=("grey-world", "white-patch"), default="grey-world")
    s.add_argument("--percentile", type=float, default=99.0); s.add_argument("-o", "--out", required=True); s.add_argument("--hist"); s.set_defaults(fn=cmd_wb)

    s = sub.add_parser("clahe", help="CLAHE по яркости"); s.add_argument("image"); s.add_argument("--clip", type=float, default=2.0); s.add_argument("--tiles", type=int, default=8)
    s.add_argument("-o", "--out", required=True); s.add_argument("--compare", help="PNG: исходный | equalizeHist | CLAHE"); s.add_argument("--hist"); s.set_defaults(fn=cmd_clahe)

    s = sub.add_parser("chromakey", help="объект по цвету на новый фон"); s.add_argument("image"); s.add_argument("--bg", required=True)
    s.add_argument("--lo", type=_ints, help="H,S,V нижняя граница"); s.add_argument("--hi", type=_ints, help="H,S,V верхняя граница")
    s.add_argument("--sample", type=_ints, help="x,y,w,h — прямоугольник на объекте: границы снимаются с него")
    s.add_argument("--open", type=int, default=5); s.add_argument("--close", type=int, default=9); s.add_argument("--feather", type=float, default=2.0)
    s.add_argument("-o", "--out", required=True); s.add_argument("--mask", help="PNG маски объекта"); s.set_defaults(fn=cmd_chromakey)

    s = sub.add_parser("rotate", help="поворот с сохранением всего кадра"); s.add_argument("image"); s.add_argument("--deg", type=float, required=True)
    s.add_argument("-o", "--out", required=True); s.set_defaults(fn=cmd_rotate)

    s = sub.add_parser("resize", help="масштабирование с антиалиасингом"); s.add_argument("image"); s.add_argument("--scale", type=float, required=True)
    s.add_argument("--method", choices=("area", "blur", "naive"), default="area"); s.add_argument("--report", action="store_true", help="PSNR трёх методов к эталону Lanczos")
    s.add_argument("-o", "--out", required=True); s.set_defaults(fn=cmd_resize)

    s = sub.add_parser("rectify", help="выпрямление по 4 углам"); s.add_argument("image")
    s.add_argument("--corners", type=lambda v: [float(t) for t in v.split(",")], help="x1,y1,x2,y2,x3,y3,x4,y4 в любом порядке")
    s.add_argument("--clicks", action="store_true", help="задать углы кликами в окне OpenCV")
    s.add_argument("--aspect", type=float, help="W / H объекта: A4 — 0.707"); s.add_argument("--size", help="WxH выхода, например 640x905")
    s.add_argument("-o", "--out", required=True); s.set_defaults(fn=cmd_rectify)

    s = sub.add_parser("sheet", help="контактный лист"); s.add_argument("images", nargs="+"); s.add_argument("--cols", type=int)
    s.add_argument("--cell-w", type=int, default=320); s.add_argument("--cell-h", type=int, default=240); s.add_argument("-o", "--out", required=True); s.set_defaults(fn=cmd_sheet)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.command == "rectify" and not a.clicks and not a.corners:
        raise SystemExit("rectify: нужны --corners x1,y1,…,x4,y4 или --clicks")
    if a.command == "chromakey" and not a.sample and not (a.lo and a.hi):
        raise SystemExit("chromakey: нужны --lo и --hi или --sample x,y,w,h")
    try:
        a.fn(a)
    except (ValueError, FileNotFoundError, cv2.error) as exc:
        raise SystemExit("%s: %s" % (a.command, exc)) from None
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None
    sys.exit(main())
