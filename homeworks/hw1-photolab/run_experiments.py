"""Повторяемый запуск CLI и сбор чисел отчёта, без заданных заранее результатов."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import PIL

import photolab as pl


ROOT = Path(__file__).resolve().parent


def run(dataset, sample=None, corners=None, aspect=210 / 297):
    course = dataset == "course"
    source = Path("sample_data" if course else "data")
    output = Path("out") / dataset
    output.mkdir(parents=True, exist_ok=True)
    dark = source / ("iu-building.jpg" if course else "dark.jpg")
    bright = source / ("ball-lamp.jpg" if course else "bright.jpg")
    obj = source / ("ball-lamp.jpg" if course else "object.jpg")
    bg = source / ("road-cam.jpg" if course else "bg.jpg")
    poster = source / ("poster-angle.jpg" if course else "poster.jpg")
    text = source / ("iu-building.jpg" if course else "text.jpg")
    paths = [dark, bright, obj, bg, poster, text]
    for path in paths:
        if not path.is_file():
            raise ValueError("нет исходной фотографии: %s" % path)
    if not course and (sample is None or corners is None):
        raise ValueError("для своих фото нужны --sample x,y,w,h и --corners x1,y1,...,x4,y4")
    records, commands = [], []

    def command(*args):
        args = [a.as_posix() if isinstance(a, Path) else str(a) for a in args]
        completed = subprocess.run([sys.executable, "photolab.py", *args], cwd=ROOT,
                                   capture_output=True, text=True, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip())
        record = json.loads(completed.stdout)
        records.append(record)
        commands.append("python photolab.py " + subprocess.list2cmdline(args))
        print(completed.stdout.strip(), flush=True)
        return record

    for value, suffix in ((0.5, "g05"), (2.0, "g20")):
        command("gamma", dark, "--gamma", value, "-o", output / f"dark_{suffix}.jpg",
                "--hist", output / f"dark_{suffix}_hist.png")
    for method, suffix in (("grey-world", "gw"), ("white-patch", "wp")):
        command("wb", bright, "--method", method, "-o", output / f"bright_{suffix}.jpg",
                "--hist", output / f"bright_{suffix}_hist.png")
    command("clahe", dark, "--clip", 2, "--tiles", 8, "-o", output / "dark_clahe.jpg",
            "--compare", output / "dark_cmp.jpg", "--hist", output / "dark_clahe_hist.png")
    bounds = ["--lo", "140,121,40", "--hi", "168,230,255"] if course else ["--sample", sample]
    command("chromakey", obj, "--bg", bg, *bounds, "-o", output / "object_on_bg.jpg",
            "--mask", output / "object_mask.png")
    if course:
        command("chromakey", source / "ball-day.jpg", "--bg", bg, "--sample", "160,530,60,60",
                "-o", output / "ball_day_on_bg.jpg", "--mask", output / "ball_day_mask.png")
    for deg in (30, -137):
        command("rotate", poster, "--deg", deg, "-o", output / f"poster_rot{deg}.jpg")
    # На здании дополнительно сверяем опубликованные размеры поворота.
    if course:
        command("rotate", dark, "--deg", 30, "-o", output / "iu_rot30.jpg")
    for method in ("naive", "blur", "area"):
        flags = ["--report"] if method == "area" else []
        command("resize", text, "--scale", 0.25, "--method", method, *flags,
                "-o", output / f"text_x025_{method}.jpg")
    command("resize", text, "--scale", 2, "-o", output / "text_x2.jpg")
    corners = "1290,1010,560,330,600,840,1250,220" if course else corners
    command("rectify", poster, "--corners", corners, "--aspect", 0.707 if course else aspect,
            "-o", output / "poster_front.jpg")
    comparison = {}
    if course:
        command("rectify", poster, "--corners", corners, "--size", "640x905",
                "-o", output / "poster_front640.png")
        mask = cv2.imread(str(output / "object_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
        yy, xx = np.indices(mask.shape)
        gt = (xx - 190) ** 2 + (yy - 560) ** 2 <= 58 ** 2
        tp = int(np.count_nonzero(mask & gt))
        comparison["ball_lamp"] = {"true_positive": tp,
            "recall_pct": round(100 * tp / np.count_nonzero(gt), 2),
            "precision_pct": round(100 * tp / max(np.count_nonzero(mask), 1), 2)}
        front = pl.imread(str(output / "poster_front640.png"))
        gt_front = pl.imread(str(source / "poster-gt.png"))
        ink = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) < 128
        comparison["poster_match_pct"] = round(100 * float((ink(front) == ink(gt_front)).mean()), 2)
    images = list(dict.fromkeys(p.as_posix() for p in paths))
    images += [p.as_posix() for p in sorted(output.glob("*.jpg")) if p.name != "sheet.jpg"]
    command("sheet", *images, "--cols", 4, "-o", output / "sheet.jpg")
    diagnostics = {}
    if not course:
        # Фиксированный участок белого листа на собственном bright.jpg.
        roi = (800, 1050, 180, 180)
        image = pl.imread(str(bright))
        x, y, w, h = roi
        diagnostics["white_paper_roi"] = {"xywh": list(roi), "means_bgr": {}}
        for method in ("original", "grey-world", "white-patch"):
            result = image if method == "original" else pl.white_balance(image, method)[0]
            diagnostics["white_paper_roi"]["means_bgr"][method] = [
                round(float(v), 1) for v in result[y:y + h, x:x + w].mean(axis=(0, 1))]
    metrics = {"dataset": dataset, "versions": {"opencv": cv2.__version__, "numpy": np.__version__,
               "pillow": PIL.__version__}, "records": records, "comparison": comparison}
    metrics["diagnostics"] = diagnostics
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "metrics.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                                         encoding="utf-8")
    (output / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(json.dumps({"comparison": comparison, "metrics": str(output / "metrics.json")}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("course", "personal"), default="course")
    parser.add_argument("--sample", help="x,y,w,h на цветной части собственного предмета")
    parser.add_argument("--corners", help="восемь координат углов собственного плаката")
    parser.add_argument("--aspect", type=float, default=210 / 297)
    args = parser.parse_args()
    # Пути и файлы результатов всегда относительно папки задания.
    import os
    os.chdir(ROOT)
    try:
        run(args.dataset, args.sample, args.corners, args.aspect)
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
