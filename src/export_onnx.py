"""Export the best scored model to ONNX (the plain file format a bodyshop PC or a phone app can run without PyTorch)
and time it on the CPU. Writes results/export.json for the results page.
Usage:  .venv/bin/python src/export_onnx.py            (picks the scored run with the highest test mask AP)
"""
import json, time
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import YAML

ROOT = Path(__file__).resolve().parents[1]
runs = {p.parent.name: json.load(open(p))["mask"]["AP"] for p in ROOT.glob("results/*/summary.json")}
best = max(runs, key=runs.get)
imgsz = YAML.load(ROOT / "runs" / best / "args.yaml")["imgsz"]
onnx_path = YOLO(ROOT / "runs" / best / "weights/best.pt").export(format="onnx", imgsz=imgsz, simplify=True, device="cpu")

# time the exported model on 30 test photos, one at a time, on the CPU only
model = YOLO(onnx_path, task="segment")
imgs = sorted((ROOT / "data/cardd_yolo/images/test").glob("*.jpg"))[:30]
model.predict(imgs[:2], device="cpu", imgsz=imgsz, verbose=False)            # warm-up, not timed
t = time.time(); model.predict(imgs, device="cpu", imgsz=imgsz, verbose=False)
ms = (time.time() - t) / len(imgs) * 1000
out = {"run": best, "imgsz": imgsz, "onnx_file": str(Path(onnx_path).relative_to(ROOT)),
       "onnx_mb": round(Path(onnx_path).stat().st_size / 1e6, 1), "ms_per_image_cpu_onnx": round(ms, 1)}
json.dump(out, open(ROOT / "results/export.json", "w"), indent=2)
print(out)
