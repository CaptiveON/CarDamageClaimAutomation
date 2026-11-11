"""COCO (CarDD) -> YOLO segmentation format. Images are symlinked, not copied.

Each image gets one .txt: one line per instance -> `cls x1 y1 x2 y2 ...` (normalised polygon).
"""
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COCO = ROOT / "Dataset/CarDD_COCO"
OUT = ROOT / "data/cardd_yolo"
NAMES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]

for split in ["train", "val", "test"]:
    d = json.load(open(COCO / f"annotations/instances_{split}2017.json"))
    assert [c["name"] for c in sorted(d["categories"], key=lambda c: c["id"])] == NAMES
    img_dir, lbl_dir = OUT / "images" / split, OUT / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True); lbl_dir.mkdir(parents=True, exist_ok=True)
    imgs = {i["id"]: i for i in d["images"]}
    lines = {i: [] for i in imgs}
    for a in d["annotations"]:
        im = imgs[a["image_id"]]; w, h = im["width"], im["height"]
        for poly in a["segmentation"]:  # CarDD: exactly one polygon per instance
            xy = [f"{v / (w if k % 2 == 0 else h):.5f}" for k, v in enumerate(poly)]
            lines[a["image_id"]].append(" ".join([str(a["category_id"] - 1)] + xy))
    for iid, im in imgs.items():
        src = COCO / f"{split}2017" / im["file_name"]; dst = img_dir / im["file_name"]
        if not dst.exists(): os.symlink(src, dst)
        (lbl_dir / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines[iid]) + "\n")
    print(f"{split}: {len(imgs)} images, {sum(map(len, lines.values()))} instances")

(OUT / "cardd.yaml").write_text(
    f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
    + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
print("wrote", OUT / "cardd.yaml")
