"""Figures for the results page, built from the saved results. Run after src/evaluate.py:
    .venv/bin/python src/figures.py
Writes results/fig_per_class.png, fig_by_size.png, fig_training.png, fig_gallery.png.
Re-run it whenever a new run has been scored; every figure picks up all runs it can find.
"""
import csv, json
from pathlib import Path
import cv2, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils import YAML

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
# Paper numbers (mask AP on the official test split), same as in evaluate.py.
PAPER = {"Mask R-CNN R-50 (paper)": (48.4, [27.0, 22.2, 9.3, 87.8, 58.5, 85.8]),
         "DCN+ R-101 (paper)":      (57.0, [40.5, 34.3, 16.6, 89.6, 70.8, 90.0])}
# Fixed order and fixed colour per run, so colours do not move when a new run is added.
LABEL = {"yolo11s-seg_640": "Ours: YOLO11s, 640 px", "yolo11s-seg_1024": "Ours: YOLO11s, 1024 px",
         "yolo26s-seg_640": "Ours: YOLO26s, 640 px", "yolo11s-seg_640_bal": "Ours: YOLO11s, 640 px + crack boost",
         "yolo26m-seg_640": "Ours: YOLO26m (medium), 640 px",
         "yolo26s-seg_640_cont": "Ours: YOLO26s, 640 px + 10 gentle epochs",
         "yolo26s-seg_640_cont2": "Ours: YOLO26s, 640 px + 20 gentle epochs",
         "yolo26s-seg_640_bal": "Ours: YOLO26s, 640 px + crack boost",
         "yolo26s-seg_1024": "Ours: YOLO26s, 1024 px from COCO"}
GREY, OURS = ["#52514e", "#898781"], ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#e34948", "#008300"]   # paper in grey, our runs in colour (up to 8 runs)
MASK_COLORS = [(255, 56, 56), (255, 200, 0), (0, 220, 255), (255, 0, 255), (0, 255, 80), (255, 140, 0)]  # same as app.py

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b"})

# scored runs, in the fixed order above
runs = {k: json.load(open(ROOT / "results" / k / "summary.json")) for k in LABEL
        if (ROOT / "results" / k / "summary.json").exists()}

# ---- 1. per-class mask AP: paper baselines vs our runs ----
series = [(n, ap, pc, GREY[i]) for i, (n, (ap, pc)) in enumerate(PAPER.items())] + \
         [(LABEL[k], d["mask"]["AP"], d["mask"]["per_class"], OURS[i]) for i, (k, d) in enumerate(runs.items())]
fig, ax = plt.subplots(figsize=(11, 4.6))
w, x = 0.8 / len(series), np.arange(6)
for i, (name, ap, vals, col) in enumerate(series):
    ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w * 0.9, color=col, label=f"{name}: overall {ap}")
ax.set_xticks(x, NAMES); ax.set_ylim(0, 100); ax.set_ylabel("mask AP (official test split)")
ax.yaxis.grid(True, color="#e1e0d9"); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left", fontsize=10)
fig.tight_layout(); fig.savefig(ROOT / "results/fig_per_class.png", dpi=200); plt.close(fig)

# ---- 2. mask AP by object size (COCO bands) ----
if runs and all("by_size" in d["mask"] for d in runs.values()):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    w, x = 0.8 / len(runs), np.arange(3)
    for i, (k, d) in enumerate(runs.items()):
        ax.bar(x + (i - (len(runs) - 1) / 2) * w, d["mask"]["by_size"], w * 0.9, color=OURS[i], label=LABEL[k])
    ax.set_xticks(x, ["small\n(< 32x32 px)", "medium", "large\n(> 96x96 px)"]); ax.set_ylim(0, 100)
    ax.set_ylabel("mask AP"); ax.yaxis.grid(True, color="#e1e0d9"); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT / "results/fig_by_size.png", dpi=200); plt.close(fig)

# ---- 3. training curves: validation mask mAP50-95 per epoch ----
fig, ax = plt.subplots(figsize=(7, 4))
for i, k in enumerate(LABEL):
    f = ROOT / "runs" / k / "results.csv"
    if k not in runs or not f.exists():     # only finished (scored) runs, not one still training
        continue
    rows = list(csv.DictReader(open(f)))
    start = 40 if k == "yolo11s-seg_1024" else 0        # only the YOLO11s 1024 run continued from the 640 run's 40 epochs
    ep = [start + int(r["epoch"]) for r in rows]; m = [float(r["metrics/mAP50-95(M)"]) * 100 for r in rows]
    ax.plot(ep, m, color=OURS[list(runs).index(k)], lw=2, label=f"{LABEL[k]}  (final {m[-1]:.1f})")   # same colour per run as the other charts
ax.set_xlabel("epoch"); ax.set_ylabel("val mask mAP50-95"); ax.set_ylim(0, 70)
ax.yaxis.grid(True, color="#e1e0d9"); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=10)
fig.tight_layout(); fig.savefig(ROOT / "results/fig_training.png", dpi=200); plt.close(fig)

# ---- 4. gallery: one test image per class, human annotation above, model prediction below ----
def draw(img, items):
    """Paint each (polygon, class, text) on a copy of the image: filled colour at 40% plus an outline and a label."""
    out, overlay = img.copy(), img.copy()
    for poly, cls, text in items:
        cv2.fillPoly(overlay, [poly], MASK_COLORS[cls])
    out = cv2.addWeighted(overlay, 0.4, out, 0.6, 0)
    for poly, cls, text in items:
        cv2.polylines(out, [poly], True, MASK_COLORS[cls], 3)
        x, y = poly[0]
        cv2.putText(out, text, (int(x), max(int(y) - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, MASK_COLORS[cls], 3)
    return out

best = max(runs, key=lambda k: runs[k]["mask"]["AP"]) if runs else None   # the best scored run, never a half-trained checkpoint
if best:
    model = YOLO(ROOT / "runs" / best / "weights/best.pt")
    imgsz = YAML.load(ROOT / "runs" / best / "args.yaml")["imgsz"]
    d = json.load(open(ROOT / "Dataset/CarDD_COCO/annotations/instances_test2017.json"))
    name_of = {i["id"]: i["file_name"] for i in d["images"]}
    gt = {}
    for a in d["annotations"]:
        gt.setdefault(a["image_id"], []).append((np.array(a["segmentation"][0], np.int32).reshape(-1, 2), a["category_id"] - 1))
    fig, axes = plt.subplots(2, 6, figsize=(16, 5.4))
    for c in range(6):
        # a tidy example: the first test image that has this class and at most 3 damages in total
        iid = next(i for i in sorted(gt) if any(cls == c for _, cls in gt[i]) and len(gt[i]) <= 3)
        img = cv2.cvtColor(cv2.imread(str(ROOT / "data/cardd_yolo/images/test" / name_of[iid])), cv2.COLOR_BGR2RGB)
        res = model.predict(img, imgsz=imgsz, conf=0.25, device="cpu", verbose=False)[0]
        pred = [] if res.masks is None else [(p.astype(np.int32), int(k), f"{NAMES[int(k)]} {cf:.2f}")
                                             for p, k, cf in zip(res.masks.xy, res.boxes.cls.tolist(), res.boxes.conf.tolist()) if len(p) >= 3]
        axes[0, c].imshow(draw(img, [(p, k, NAMES[k]) for p, k in gt[iid]])); axes[0, c].set_title(NAMES[c], fontsize=13)
        axes[1, c].imshow(draw(img, pred))
        if c == 2:   # keep one big example for the results page (the crack photo: crack plus a scratch)
            cv2.imwrite(str(ROOT / "results/fig_example.png"), cv2.cvtColor(draw(img, pred), cv2.COLOR_RGB2BGR))
        for r in range(2):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            for s in axes[r, c].spines.values(): s.set_visible(False)
    axes[0, 0].set_ylabel("human annotation", fontsize=12); axes[1, 0].set_ylabel("model prediction", fontsize=12)
    fig.tight_layout(); fig.savefig(ROOT / "results/fig_gallery.png", dpi=150); plt.close(fig)

print("figures written to results/: ", sorted(p.name for p in ROOT.glob("results/fig_*.png")))
