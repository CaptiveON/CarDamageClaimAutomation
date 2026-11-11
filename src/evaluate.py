"""Score a trained model on the official CarDD test split (374 images) exactly the way the paper did:
COCO Average Precision computed by pycocotools. This lets us put our numbers in the same table as the paper.

Usage:  .venv/bin/python src/evaluate.py runs/yolo26s-seg_640 [cpu|mps]   (default mps)
Writes: results/<run name>/summary.json and summary.md (plus confusion matrix and curves from Ultralytics)
"""
import json, sys, time
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import YAML
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / "Dataset/CarDD_COCO/annotations/instances_test2017.json"
NAMES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
# Numbers copied from the CarDD paper (Tables IV and V, test split) so the comparison sits in one table.
PAPER = {
    "Mask R-CNN R-50 (paper)": {"mask": {"AP": 48.4, "AP50": 65.5, "per_class": [27.0, 22.2, 9.3, 87.8, 58.5, 85.8]},
                                "box":  {"AP": 49.2, "AP50": 66.3, "per_class": [27.0, 25.3, 19.5, 87.0, 53.8, 82.6]}},
    "DCN+ R-101 (paper)":      {"mask": {"AP": 57.0, "AP50": 77.7, "per_class": [40.5, 34.3, 16.6, 89.6, 70.8, 90.0]},
                                "box":  {"AP": 60.6, "AP50": 78.8, "per_class": [42.2, 42.3, 29.6, 90.1, 69.5, 90.2]}},
}

run = Path(sys.argv[1])
DEVICE = sys.argv[2] if len(sys.argv) > 2 else "mps"   # use cpu while the GPU is busy training
# Note: Ultralytics ignores augment=True (test-time augmentation) for segmentation models, so there is no TTA option here.
imgsz = YAML.load(run / "args.yaml")["imgsz"]      # use the same picture size the model was trained at
model = YOLO(run / "weights/best.pt")
out_dir = ROOT / "results" / run.name
out_dir.mkdir(parents=True, exist_ok=True)

# 1) Run the model on the test images. save_json=True writes predictions.json in COCO format.
r = model.val(data=str(ROOT / "data/cardd_yolo/cardd.yaml"), split="test", imgsz=imgsz, batch=8, device=DEVICE,
              save_json=True, plots=True, project=str(ROOT / "results"), name=run.name, exist_ok=True, verbose=False)
speed = sum(r.speed.values())                                    # ms per image (pre + inference + post)

# 2) Score the predictions with pycocotools, the official COCO scorer the paper used.
gt = COCO(str(GT))
dt = gt.loadRes(str(Path(r.save_dir) / "predictions.json"))
summary = {"run": run.name, "imgsz": imgsz, f"ms_per_image_{DEVICE}": round(speed, 1)}
for kind, key in [("bbox", "box"), ("segm", "mask")]:
    ev = COCOeval(gt, dt, kind); ev.evaluate(); ev.accumulate(); ev.summarize()
    p = ev.eval["precision"]        # shape: [IoU thresholds, recall steps, class, object size, max detections]
    per_class = []
    for k in range(len(NAMES)):
        v = p[:, :, k, 0, -1]        # all sizes, 100 detections per image
        per_class.append(round(float(v[v > -1].mean()) * 100, 1))  # -1 means "no ground truth", ignore it
    summary[key] = {"AP": round(ev.stats[0] * 100, 1), "AP50": round(ev.stats[1] * 100, 1),
                    "AP75": round(ev.stats[2] * 100, 1), "per_class": per_class,
                        "by_size": [round(ev.stats[i] * 100, 1) for i in (3, 4, 5)]}  # small, medium, large objects
assert len(summary["mask"]["per_class"]) == 6 and 0 <= summary["mask"]["AP"] <= 100

# 3) Speed on a plain CPU (what a bodyshop PC would get): average over 30 test images.
imgs = sorted((ROOT / "data/cardd_yolo/images/test").glob("*.jpg"))[:30]
model.predict(imgs[:2], device="cpu", imgsz=imgsz, verbose=False)   # warm-up, not timed
t = time.time(); model.predict(imgs, device="cpu", imgsz=imgsz, verbose=False)
summary["ms_per_image_cpu"] = round((time.time() - t) / len(imgs) * 1000, 1)

# 4) Save a json and a markdown table for the results page.
json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
rows = [f"| {n} | {d['mask']['AP']} | {d['mask']['AP50']} | " + " | ".join(map(str, d['mask']['per_class'])) + " |"
        for n, d in PAPER.items()]
rows.append(f"| **{run.name} (ours, {imgsz}px)** | **{summary['mask']['AP']}** | {summary['mask']['AP50']} | "
            + " | ".join(map(str, summary['mask']['per_class'])) + " |")
md = ("| Model | mask AP | mask AP50 | " + " | ".join(NAMES) + " |\n|---|---|---|" + "---|" * 6 + "\n" + "\n".join(rows)
      + f"\n\nBox AP {summary['box']['AP']} (DCN+ 60.6, Mask R-CNN 49.2). "
      f"Speed: {speed:.1f} ms/image on {DEVICE} (batches of 8), {summary['ms_per_image_cpu']} ms for a single image on CPU.\n")
(out_dir / "summary.md").write_text(md)
print(md)
