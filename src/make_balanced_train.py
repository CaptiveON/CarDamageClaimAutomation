"""Make a second training set in which every photo that contains a crack appears three times instead of once.
Cracks are only 7% of the labels, so the model rarely sees one; showing those photos more often is the simplest fix.
Writes data/cardd_yolo/images/train_bal + labels/train_bal (symlinks and label copies) and cardd_bal.yaml.
Run:  .venv/bin/python src/make_balanced_train.py
"""
import os, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/cardd_yolo"
img_dir, lbl_dir = D / "images/train_bal", D / "labels/train_bal"
img_dir.mkdir(parents=True, exist_ok=True); lbl_dir.mkdir(parents=True, exist_ok=True)
count = {"photos in the new set": 0, "photos with a crack": 0}
for lbl in sorted((D / "labels/train").glob("*.txt")):
    has_crack = any(line.split()[0] == "2" for line in lbl.read_text().splitlines() if line.strip())  # class 2 = crack
    for k in range(3 if has_crack else 1):
        stem = lbl.stem if k == 0 else f"{lbl.stem}_dup{k}"
        dst = img_dir / f"{stem}.jpg"
        if not dst.exists():
            os.symlink((D / "images/train" / f"{lbl.stem}.jpg").resolve(), dst)
        shutil.copy(lbl, lbl_dir / f"{stem}.txt")
        count["photos in the new set"] += 1
    count["photos with a crack"] += has_crack
print(count)
(D / "cardd_bal.yaml").write_text((D / "cardd.yaml").read_text().replace("train: images/train\n", "train: images/train_bal\n"))
print("wrote", D / "cardd_bal.yaml")
