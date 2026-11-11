"""Does the parts model work on CarDD close-ups? Run it on N official test photos, join the human damage outlines to the
predicted parts with the demo's own rule, and count how often a damage lands on a named panel. Writes
results/parts_yolo26s_640/cardd_check_<conf>.json (the numbers the research page quotes) and a contact sheet .jpg beside it
(CarDD photos, so it stays on the laptop: results/*/*.jpg is never uploaded).
Usage: .venv/bin/python src/parts_check.py [part confidence=0.6] [N photos=100] [device=cpu]
"""
import json, random, sys
from collections import Counter
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from panels import detect_parts, name_panels
from ultralytics import YOLO

conf = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
RUN = "parts_yolo26s_640"
weights = ROOT / "weights_trained" / RUN / "best.pt"
sheet = ROOT / "results" / RUN / f"cardd_check_{conf}.jpg"
NAMES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]

gt = json.load(open(ROOT / "Dataset/CarDD_COCO/annotations/instances_test2017.json"))
by_img = {}
for a in gt["annotations"]:
    by_img.setdefault(a["image_id"], []).append(a)
imgs = {i["id"]: i["file_name"] for i in gt["images"]}
random.seed(0)
picks = random.sample(sorted(by_img), N)
model = YOLO(weights)

total, named, sided, per_class = Counter(), Counter(), Counter(), {}
no_parts, n_parts, tiles = 0, [], []
for k, iid in enumerate(picks):
    bgr = cv2.imread(str(ROOT / "data/cardd_yolo/images/test" / imgs[iid])); h, w = bgr.shape[:2]
    parts = detect_parts(model, bgr, device, conf)
    n_parts.append(len(parts)); no_parts += not parts
    polys = [np.array(a["segmentation"][0], dtype=np.int32).reshape(-1, 2) for a in by_img[iid]]
    res = name_panels(polys, parts, w, h)
    for a, (label, share, _) in zip(by_img[iid], res):
        c = NAMES[a["category_id"] - 1]
        total[c] += 1
        if label != "unknown":
            named[c] += 1
            if not label.endswith("side unknown"):
                sided[c] += 1
    if k < 12:   # contact sheet: parts in white, human damage outlines in red with the panel they were given
        im = bgr.copy()
        for p in parts:
            cv2.polylines(im, [p["poly"]], True, (245, 245, 245), 2)
            x, y = p["poly"].mean(0).astype(int)
            cv2.putText(im, p["name"] + (f" {p['side']}" if p["side"] else ""), (int(x) - 40, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2)
        for poly, (label, share, _) in zip(polys, res):
            cv2.polylines(im, [poly], True, (0, 0, 255), 3)
            x, y = poly[0]
            cv2.putText(im, label + (f" {share}%" if share else ""), (int(x), max(int(y) - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        s = 480 / w; im = cv2.resize(im, (480, int(h * s)))
        canvas = np.full((360, 480, 3), 30, np.uint8); canvas[:min(360, im.shape[0])] = im[:360]
        tiles.append(canvas)
rows = [np.hstack(tiles[i:i + 4]) for i in range(0, 12, 4)]
cv2.imwrite(str(sheet), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])

T, Nd, S = sum(total.values()), sum(named.values()), sum(sided.values())
print(f"{N} test photos, {T} human-labelled damages. Photos with no part found: {no_parts}. Parts per photo: median {int(np.median(n_parts))}, max {max(n_parts)}")
print(f"damage on a named panel: {Nd}/{T} = {100*Nd/T:.0f}%   with a side where one applies or none needed: {S}/{T} = {100*S/T:.0f}%")
for c in NAMES:
    if total[c]:
        print(f"  {c:14} {named[c]:3}/{total[c]:3} named ({100*named[c]/total[c]:.0f}%)")
print("contact sheet:", sheet)
out = {"weights": str(weights.relative_to(ROOT)), "part_confidence": conf, "photos": N, "damages": T, "named": Nd,
       "named_pct": round(100 * Nd / T, 1), "photos_without_parts": no_parts,
       "named_per_class": {c: [named[c], total[c]] for c in NAMES if total[c]}}
json.dump(out, open(ROOT / "results" / RUN / f"cardd_check_{conf}.json", "w"), indent=2)
