"""One picture that explains the dataset: how many of each damage type, how many are tiny, and examples.
Run:  .venv/bin/python src/dataset_figure.py   ->  results/dataset_overview.png
"""
import json, random, collections
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
d = json.load(open(ROOT / "Dataset/CarDD_COCO/annotations/instances_train2017.json"))
NAMES = [c["name"] for c in sorted(d["categories"], key=lambda c: c["id"])]
COLORS = ["#ff3838", "#ffc800", "#00dcff", "#ff00ff", "#00ff50", "#ff8c00"]

count = collections.Counter(a["category_id"] for a in d["annotations"])
small = collections.Counter(a["category_id"] for a in d["annotations"] if a["area"] < 128 ** 2)  # paper's "small" rule
fig = plt.figure(figsize=(16, 9))
grid = fig.add_gridspec(2, 6, height_ratios=[1, 1.1])

ax = fig.add_subplot(grid[0, :3])
ax.bar(NAMES, [count[i] for i in range(1, 7)], color=COLORS)
ax.set_title("Training instances per damage type (11:1 imbalance)"); ax.tick_params(axis="x", rotation=20)
for i in range(6): ax.text(i, count[i + 1] + 30, str(count[i + 1]), ha="center")

ax = fig.add_subplot(grid[0, 3:])
ax.bar(NAMES, [100 * small[i] / count[i] for i in range(1, 7)], color=COLORS)
ax.set_title("Share of instances that are small (< 128x128 px)"); ax.set_ylabel("%"); ax.tick_params(axis="x", rotation=20)
for i in range(6): ax.text(i, 100 * small[i + 1] / count[i + 1] + 1, f"{100 * small[i + 1] / count[i + 1]:.0f}%", ha="center")

# one example picture per class, with its masks drawn
random.seed(7)
imgs = {i["id"]: i for i in d["images"]}
by_img = collections.defaultdict(list)
for a in d["annotations"]: by_img[a["image_id"]].append(a)
for k, cid in enumerate(range(1, 7)):
    ax = fig.add_subplot(grid[1, k])
    iid = random.choice([i for i, anns in by_img.items() if any(a["category_id"] == cid for a in anns) and len(anns) <= 3])
    ax.imshow(Image.open(ROOT / "Dataset/CarDD_COCO/train2017" / imgs[iid]["file_name"])); ax.axis("off")
    for a in by_img[iid]:
        pts = list(zip(a["segmentation"][0][0::2], a["segmentation"][0][1::2]))
        ax.add_patch(Polygon(pts, fill=True, alpha=0.35, color=COLORS[a["category_id"] - 1]))
        ax.add_patch(Polygon(pts, fill=False, lw=1.5, color=COLORS[a["category_id"] - 1]))
    ax.set_title(", ".join(sorted({NAMES[a["category_id"] - 1] for a in by_img[iid]})), fontsize=10)
plt.tight_layout(); plt.savefig(ROOT / "results/dataset_overview.png", dpi=110); print("saved results/dataset_overview.png")
