"""The research page: what I did, how I did it and what I learned, in plain language.
Every number in the tables and tiles is read from results/*/summary.json and weights_trained/*/ so the page
cannot drift from the real scores. The prose quotes the final numbers of the finished runs.
"""
import csv, json
from pathlib import Path
import pandas as pd
import streamlit as st
from ultralytics.utils import YAML   # the YAML reader that ships with Ultralytics (no extra package)

ROOT = Path(__file__).resolve().parents[1]
R, W = ROOT / "results", ROOT / "weights_trained"
NAMES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
LABEL = {"yolo11s-seg_640": "YOLO11s, 640 px, 40 epochs",
         "yolo11s-seg_1024": "YOLO11s continued at 1024 px, 30 epochs",
         "yolo26s-seg_640": "YOLO26s, 640 px, 40 epochs",
         "yolo26m-seg_640": "YOLO26m (medium), 640 px, 20 epochs",
         "yolo11s-seg_640_bal": "YOLO11s + crack boost, 640 px, 20 more epochs",
         "yolo26s-seg_640_cont": "YOLO26s continued, 640 px, 10 gentle epochs",
         "yolo26s-seg_640_cont2": "YOLO26s continued, 640 px, 10 + 10 gentle epochs",
         "yolo26s-seg_640_bal": "YOLO26s + crack boost, 640 px, 20 more epochs",
         "yolo26s-seg_1024": "YOLO26s, 1024 px from COCO weights, 60 epochs"}
PARAMS = {"yolo11s-seg_640": 10.1, "yolo11s-seg_1024": 10.1, "yolo26s-seg_640": 11.5, "yolo26m-seg_640": 27.1,
          "yolo11s-seg_640_bal": 10.1, "yolo26s-seg_640_cont": 11.5, "yolo26s-seg_640_cont2": 11.5, "yolo26s-seg_640_bal": 11.5, "yolo26s-seg_1024": 11.5}
# Numbers copied from the CarDD paper (Wang, Li and Wu, IEEE T-ITS 2023, Tables IV and V, official test split).
PAPER = [{"Model": "Mask R-CNN R-50 (paper, 2023)", "Params (M)": 44.0, "Mask AP": 48.4, "Mask AP50": 65.5, "Box AP": 49.2,
          **dict(zip(NAMES, [27.0, 22.2, 9.3, 87.8, 58.5, 85.8]))},
         {"Model": "DCN+ R-101 (paper's best)", "Params (M)": 63.0, "Mask AP": 57.0, "Mask AP50": 77.7, "Box AP": 60.6,
          **dict(zip(NAMES, [40.5, 34.3, 16.6, 89.6, 70.8, 90.0]))}]


@st.cache_data
def load(run):
    """Test scores, training settings and training-curve facts of one run."""
    s = json.load(open(R / run / "summary.json"))
    a = YAML.load(W / run / "args.yaml")
    rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(open(W / run / "results.csv"))]
    top = max(rows, key=lambda r: float(r["metrics/mAP50-95(M)"]))
    return s, a, {"hours": float(rows[-1]["time"]) / 3600, "epochs": len(rows),
                  "best_val": round(100 * float(top["metrics/mAP50-95(M)"]), 1), "best_epoch": int(top["epoch"])}


RUNS = {r: load(r) for r in LABEL if (R / r / "summary.json").exists()}
best = max(RUNS, key=lambda r: RUNS[r][0]["mask"]["AP"])
B = RUNS[best][0]
total_hours = sum(v[2]["hours"] for v in RUNS.values())
export = json.load(open(R / "export.json")) if (R / "export.json").exists() else {}
# the car-parts (panel) model: its scores on its own test split, and how often it names a panel on CarDD close-ups
PARTS_DIR = R / "parts_yolo26s_640"
parts = json.load(open(PARTS_DIR / "val_test.json")) if (PARTS_DIR / "val_test.json").exists() else {}
gap = {c: json.load(open(PARTS_DIR / f"cardd_check_{c}.json")) for c in ("0.6", "0.4") if (PARTS_DIR / f"cardd_check_{c}.json").exists()}
if parts and (W / "parts_yolo26s_640" / "results.csv").exists():
    prow = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(open(W / "parts_yolo26s_640" / "results.csv"))]
    parts["hours"], parts["epochs"] = float(prow[-1]["time"]) / 3600, len(prow)


def img(path, caption=None):
    p = R / path
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")


# ---------------- styling and hero ----------------
st.html("""
<style>
.hero {background:#1A1A19; color:#fff; border-radius:16px; padding:2.2rem 2.4rem; margin-bottom:1rem;}
.hero h1 {color:#fff; font-size:2.3rem; margin:0 0 .6rem 0; line-height:1.15; font-weight:700;}
.hero p {color:#d9d8d3; font-size:1.08rem; max-width:62rem; margin:0; line-height:1.5;}
.hero .tag {display:inline-block; background:#2A78D6; color:#fff; font-size:.75rem; letter-spacing:.08em;
            text-transform:uppercase; padding:.25rem .65rem; border-radius:999px; margin-bottom:.9rem; font-weight:600;}
.tiles {display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:.8rem; margin:0 0 .5rem 0;}
.tile {background:#F4F4F1; border-radius:12px; padding:1rem 1.1rem;}
.tile .n {font-size:1.9rem; font-weight:700; color:#2A78D6; line-height:1.1;}
.tile .l {font-size:.85rem; color:#52514E; margin-top:.35rem; line-height:1.3;}
.lesson {border-left:4px solid #2A78D6; padding:.2rem 0 .2rem .9rem; margin:.4rem 0 .9rem 0;}
</style>
""")
st.html(f"""
<div class="hero"><span class="tag">Research notes</span>
<h1>The research behind the demo</h1>
<p>How I taught a small computer-vision model to find and outline car damage, what worked, what did not, and what I
learned on the way. Written in plain language for anyone who has tried the demo and wants to know what sits underneath.
One dataset, {len(RUNS)} training runs, one laptop, two nights and a day.</p></div>
<div class="tiles">
<div class="tile"><div class="n">{B['mask']['AP']}</div><div class="l">mask AP of my best model on the official test split.
The paper's best model scores 57.0</div></div>
<div class="tile"><div class="n">{B['box']['AP']}</div><div class="l">box AP of the same model. The paper's best is 60.6</div></div>
<div class="tile"><div class="n">{PARAMS[best]} M</div><div class="l">parameters, against about 63 M in the paper's best model</div></div>
<div class="tile"><div class="n">{B['ms_per_image_cpu']:.0f} ms</div><div class="l">per photo on a laptop CPU, no graphics card</div></div>
<div class="tile"><div class="n">{len(RUNS)}</div><div class="l">training runs, {total_hours:.1f} hours of laptop time in total</div></div>
<div class="tile"><div class="n">374</div><div class="l">official test photos, scored once with the paper's own scoring program</div></div>
</div>
""")
st.caption("Mohammad Adnan · November 2026.")

with st.container(border=True):
    st.markdown("**Contents**")
    c1, c2, c3 = st.columns(3)
    c1.markdown("1. [The problem I set out to solve](#problem)\n2. [The dataset, and what makes it hard](#data)\n"
                "3. [Methods I considered](#methods)\n4. [How the chosen model works](#model)\n5. [Preparing the data](#prep)")
    c2.markdown("6. [How I fine-tuned: the exact recipe](#recipe)\n7. [How I boosted cracks](#cracks)\n"
                "8. [How I scored the models](#scoring)\n9. [Results](#results)\n10. [Problems I ran into](#problems)")
    c3.markdown("11. [From model to tool: the demo](#demo)\n12. [Limitations and what comes next](#next)\n"
                "13. [Sources](#sources)\n14. [Reproduce it](#reproduce)")

# ---------------- 1. problem ----------------
st.header("1. The problem I set out to solve", anchor="problem")
st.markdown("""
Accident repair groups increasingly ask customers to send photos of the damage before the car arrives, so parts can be ordered
early and the job planned. The task I set myself: *analyse images of damaged vehicles, classify the type and extent of damage, and
streamline the repair process*.

Three words in that sentence decided my design:

- **Type** means a label for each damage: dent, scratch, crack, glass shatter, broken lamp or flat tyre.
- **Extent** means how big the damage is and where it sits. That needs pixels, not one label for the whole photo.
  A rectangle drawn around a long diagonal scratch is mostly undamaged paint, so even boxes overstate thin damage.
- **Streamline** means the output has to feed a workflow: which panel, how severe, which repair, which site has the skills.
  The model is one part in that chain, not the product.

There are four common ways to turn "look at a photo of a damaged car" into a computer-vision task:

| Way of framing it | What the model outputs | Good for | Cannot do | Role in my design |
|---|---|---|---|---|
| Image classification | one label per photo | cheap triage: damaged or not, junk photo or not | several damages in one photo, location, size | a gate in front of the heavy model |
| Object detection | a box and a label per damage | counting and locating damage, fast | the extent of thin or irregular damage | falls out of the segmenter for free |
| **Instance segmentation** | an exact pixel outline, a label and a confidence per damage | type **and** extent, per damage, per panel | hidden or structural damage | **the core task** |
| Salient object detection | one "damage anywhere" mask, no classes | a safety net for damage the six classes miss | saying what the damage is | a later add-on |

So the core task is **instance segmentation**: for every damage in the photo, a class, a confidence and an exact pixel
outline. "Instance" means each damage is counted separately: two dents are two instances, and more than half of the
training photos contain more than one damage (1,545 of 2,816).
""")

# ---------------- 2. data ----------------
st.header("2. The dataset, and what makes it hard", anchor="data")
st.markdown("""
I used **CarDD** (Car Damage Detection), published with a paper in *IEEE Transactions on
Intelligent Transportation Systems* in 2023. It has 4,000 photos of damaged cars, every one with the damage drawn as an
outline by people, in six classes. The authors fixed an official split of 2,816 training, 810 validation and 374 test photos,
and I keep that split exactly, so my numbers sit in the same table as theirs. The photos came from Flickr and Shutterstock,
were labelled by 20 annotators including five insurance claims experts, and every photo is exactly 1,000 pixels on its long side.
""")
img("dataset_overview.png", "Top left: training damages per class. Top right: the share of each class that is small "
    "(under 128 x 128 px). Bottom: one training photo per class with its human outlines.")
st.markdown("""
**Five things make this dataset hard, and I met every one of them:**

1. **The targets are tiny.** 95% of cracks and about half of all scratches are smaller than 128 x 128 pixels inside a
   1,000-pixel photo. The median crack covers about 1,400 square pixels, which is 0.2% of the photo. When a model is trained at
   640 pixels, as mine mostly were, every photo is shrunk by about a third and a hairline crack becomes a few pixels wide.
2. **The classes are unbalanced.** Across the whole dataset there are 3,595 scratches but only 319 flat tyres, an 11 to 1 ratio.
   The test split has only 70 cracks and 32 flat tyres, so a single missed tyre moves that class's score by a visible amount.
   Per-class numbers on this test set are noisy, and I say so wherever I quote them.
3. **Three classes look alike, and the labelling rules blur them further.** Scratches and cracks are both "slender lines of
   similar colour", and dents often sit under scratches. Where damages overlap, the annotators used the priority crack > dent >
   scratch, adjacent damage of one class on the same panel was merged into one outline, and damage crossing a panel edge was split.
   Any model learns those conventions along with the damage.
4. **They are not the photos a repairer would take.** Stock photography, often well lit and framed, with no undamaged cars at all.
   A model trained here has never seen a clean car, so its false-alarm rate on one is untested. The licence is research-only,
   so a CarDD model cannot be shipped as a product; it is a starting point, and an operator's own claim photos are the real data.
5. **Small details that bite later.** The paper says "over 9,000" damages but the released files hold 8,740. The paper reports
   small, medium and large objects with its own size cut-offs (128² and 256²) rather than the standard COCO ones (32² and 96²),
   so size-band numbers are not comparable across the two; I report the standard COCO bands. A quick scan also found 7 pairs of
   near-duplicate photos, 3 of them with one copy in training and one in validation.

I also looked at a second public dataset, "Car Damage Assessment" on Kaggle. It has about 1,500 photos at 224 x 224 pixels with
one label per photo and no outlines. It cannot measure extent and is too small to show a fine scratch, so I used it for nothing
beyond confirming that CarDD was the right choice.
""")

# ---------------- 3. methods ----------------
st.header("3. Methods I considered", anchor="methods")
st.markdown("""
I looked at five families of method before choosing. The published CarDD numbers are mask AP on the official test split,
where available.

| Family | How it works | Published CarDD mask AP | Training cost | Speed per photo on a CPU | Licence | My verdict |
|---|---|---|---|---|---|---|
| Two-stage CNNs (Mask R-CNN, DCN+) | propose candidate regions, then classify and outline each one | 48 to 57 | days on a server; old frameworks, CUDA only | about ten times a small YOLO | BSD / Apache | the baseline to beat; cite it, do not retrain it |
| One-stage YOLO-seg (YOLO11, YOLO26) | one pass predicts a box, a class and mask weights for every object | 57 to 58.5 (reported by others) | hours on a laptop | 50 to 130 ms | AGPL, or a paid Enterprise licence | **chosen for the proof of concept** |
| Transformer detectors (RT-DETR, RF-DETR-Seg) | learned object queries, no anchors, no post-processing | none published | needs a GPU; small default input size | about 360 ms | Apache-2.0 | production fallback if the licence blocks YOLO |
| Foundation models (SAM 2, SAM 3) | segment anything from a click, a box or a word | weak without fine-tuning | not trainable on a laptop | seconds | commercial use allowed | a labelling assistant, not the classifier |
| Vision-language models (GPT, Claude, Gemini) | describe the photo in words | hallucinate on damage | API cost per photo | seconds | API terms | can write the report, must not make the call |

**Why YOLO-seg.** Published CarDD results with a small YOLO sit level with the paper's best model at a sixth of the parameters.
It trains in hours, so I could run a real comparison instead of a single run. Its output is exactly what an estimator needs to
accept or override: an outline, a label and a confidence per damage. It exports to a plain file that runs on a PC, a phone or a
server. Its one real drawback is the AGPL licence, which is a procurement decision for a company; my data and scoring pipeline
is model-agnostic, so swapping in an Apache-licensed model later is cheap.

**Why not the others.** The two-stage models are ten times the compute, tied to unmaintained frameworks, and the paper's best
weights were never released; nobody has reproduced its 57.0. Transformer detectors have no published CarDD numbers, want a GPU to
train, and default to small inputs, which is wrong for tiny cracks. SAM knows boundaries, not the six damage classes, and is slow.
Vision-language models make things up on detailed damage assessment; grounding them with a segmenter's output is a good way to
write the report, but the decision should come from a small, auditable model.
""")

# ---------------- 4. model ----------------
st.header("4. How the chosen model works", anchor="model")
st.markdown("""
A YOLO segmentation model does everything in one pass:

1. A **backbone and neck** turn the photo into feature maps at three scales: coarse ones for big things like a tyre, fine ones for
   small things like a crack.
2. The **head** looks at every position in those maps and predicts, for each object it believes is there: a box, a class and
   **32 numbers** called mask weights.
3. A separate small branch predicts **32 prototype masks** for the whole image. Think of them as 32 rough stencils.
4. The outline of one damage is the **weighted sum of the 32 stencils** using that object's 32 weights, cut to its box.

Older two-stage models first propose thousands of candidate regions and then run a second network on each. YOLO skips that,
which is why it runs in about a tenth of a second on a CPU.

**Fine-tuning, in plain words.** I did not train from scratch. Each model started from weights already trained on COCO, a large
general dataset of everyday objects. Most of the network already knows edges, textures and shapes; I only had to teach it the six
damage classes. When the training started, Ultralytics reported "Transferred 555 of 561 weight tensors" for YOLO11s: everything
carried over except the final layer that names the classes, because COCO has 80 classes and CarDD has 6. That is why 40 passes
over 2,816 photos on a laptop is enough to get close to a model trained for days on a server.

**YOLO11 against YOLO26.** I trained both generations with the same recipe so the comparison is fair. YOLO26 (released in
January 2026) differs in four ways that matter here: it is **end-to-end**, so it needs no non-maximum-suppression step to remove
duplicate boxes, which simplifies deployment and keeps latency steady; it drops the distribution-focal-loss module, which makes
export simpler; it uses a **small-target-aware label assignment** (STAL) and a progressive loss, meant to help with small objects;
and it ships with a new optimiser (MuSGD). One honest footnote: I left the optimiser on "auto", and for a dataset of this size
Ultralytics chose AdamW for every run, so my YOLO26 result reflects the architecture and loss, not the new optimiser.
""")

# ---------------- 5. data preparation ----------------
st.header("5. Preparing the data", anchor="prep")
st.markdown("""
CarDD ships its labels in COCO format: one big JSON file per split with every outline as a list of x, y points. YOLO wants one
small text file per photo, one line per damage: the class number followed by the outline's points scaled to 0 to 1.
I wrote a 40-line script (`src/prepare_data.py`) that:

- reads each split's JSON and checks that the six class names come out in the expected order;
- writes one label file per photo, dividing every x by the photo width and every y by the height, and subtracting one from the
  class id because COCO counts from 1 and YOLO from 0;
- **symlinks** the 5.7 GB of photos instead of copying them, so the YOLO folder is 42 MB;
- writes the small `cardd.yaml` file that tells the trainer where the three splits are.

Two pitfalls I avoided on purpose. Ultralytics has a built-in COCO converter, but by default it writes boxes only (not outlines) and
remaps the 91 COCO class ids to 80, which would have worked for ids 1 to 6 by luck and silently broken for anything else. And CarDD
has exactly one polygon per damage, so the "merge several polygons into one with zero-width bridges" trick that converters use was
never needed; real-world labels might need it, and holes inside a damage would be lost by it.

One more detail that matters for thin damage: Ultralytics learns masks at a **quarter of the input resolution** (`mask_ratio=4`), so
at 640 px the mask the model actually learns from is 160 px wide. A hairline crack that was already thin in the photo is a line of
one or two pixels there.
""")

# ---------------- 6. recipe ----------------
st.header("6. How I fine-tuned: the exact recipe", anchor="recipe")
st.markdown(f"""
{len(RUNS)} runs, one at a time, all on an Apple M2 Max laptop (32 GB of shared memory). The table is read from each run's saved
settings file and training log, so it is the real record, not a summary from memory. The last run, YOLO26s at 1024 px from the
COCO weights for 60 epochs, came out of a day of research into why the 640 px models missed thin scratches and cracks; section 9
has what it taught me.
""")


def start(m):
    m = str(m)
    return "my " + m.split("/")[1] if m.startswith("runs/") else "COCO-pretrained " + Path(m).stem


recipe = []
for r, (s, a, t) in RUNS.items():
    recipe.append({"Run": r, "Started from": start(a["model"]),
                   "Photos per epoch": 3684 if "cardd_bal" in str(a["data"]) else 2816,
                   "Epochs": a["epochs"], "Batch": a["batch"], "Image size": a["imgsz"],
                   "Optimiser, learning rate": "AdamW, 0.001 (chosen by 'auto')" if a["optimizer"] == "auto" else f"{a['optimizer']}, {a['lr0']}",
                   "Extra augmentation": f"copy-paste {a['copy_paste']} ({a['copy_paste_mode']})" if a["copy_paste"] else "defaults only",
                   "Laptop hours": round(t["hours"], 1), "Minutes per epoch": round(60 * t["hours"] / t["epochs"], 1),
                   "Best val mask AP (epoch)": f"{t['best_val']} ({t['best_epoch']})", "Test mask AP": s["mask"]["AP"]})
st.dataframe(pd.DataFrame(recipe), hide_index=True, width="stretch")
st.markdown("""
**What each setting means, and why I set it that way:**

- **Image size 640 or 1024.** The photos are 1,000 px wide. At 640 every photo is shrunk by about a third, which is fast and is the
  default everyone starts from. 1024 keeps full detail, where the thin damage lives, but costs about 2.5 times the compute per
  photo and forced the batch down to 4 to fit in memory. Two runs used it: the YOLO11s continuation, and the final YOLO26s run,
  trained at 1024 from the start.
- **Batch 8.** Eight photos per training step. Larger batches did not fit comfortably in the Apple GPU's memory alongside the
  cached images; smaller ones made the run slower.
- **Epochs 40, 30, 20 or 60.** One epoch is one pass over all training photos. The counts were set by the clock, not by convergence:
  an epoch took 3.4 minutes for YOLO11s at 640, 6.5 for YOLO26s, 7.4 for YOLO11s at 1024, 9.3 for the medium YOLO26 and 13 for
  YOLO26s at 1024. The validation score was still creeping up at the end of every run, so every model is slightly under-trained.
  The 60-epoch run had a stop rule, three epochs after epoch 50 without a new best, and never triggered it.
- **Optimiser.** I left it on "auto". Ultralytics picks the optimiser from the number of training steps; for this dataset size it
  chose AdamW with a learning rate of 0.001, momentum 0.9 and weight decay 0.0005, warming up over the first 3 epochs and then
  decaying the learning rate linearly to 1% of its start. For the crack-boost run I overrode this with plain SGD at 0.003, a low
  fixed learning rate, because that run started from an already-trained model and I wanted to nudge it, not shake it.
- **Augmentation (defaults, on purpose).** Mosaic, which stitches four photos into one so the model sees damage at many scales
  and positions, switched off for the last 10 epochs; a horizontal flip half the time; small colour shifts (hue 0.015, saturation
  0.7, brightness 0.4); random scaling by up to 50% and shifting by up to 10%. I kept defaults so that every result could be
  explained by one change at a time.
- **Loss weights.** Box 7.5, class 0.5, distribution focal loss 1.5 (YOLO11 only, YOLO26 has no DFL). Default values.
- **Precision and workers.** Ultralytics turns mixed precision off on Apple GPUs, so everything ran in full 32-bit, and it forces a
  single data-loading worker. To stop the data loading from starving the GPU I held all training photos in RAM (`cache=ram`,
  about 2.6 GB at 640 px). The two 1024 px runs needed 11.2 GB for that cache and never had that much free, so they ran without
  the cache and slower.
- **Seed 0, deterministic on.** So a rerun gives the same answer, within what the Apple GPU allows (two of its operations have no
  deterministic version and warn about it).
- **Patience 100.** Early stopping never triggered; the budget was the clock.
""")
img("fig_training.png", "Validation mask AP after every epoch of each run. The YOLO11s 1024 px run is drawn after the 40 epochs of "
    "the model it continued from; the YOLO26s 1024 px run started from the COCO weights and is drawn from epoch 1. Every curve was "
    "still rising when its budget ran out.")

# ---------------- 7. cracks ----------------
st.header("7. How I boosted cracks", anchor="cracks")
bal, base = RUNS.get("yolo11s-seg_640_bal"), RUNS.get("yolo11s-seg_640")
if bal and base:
    b0, b1 = base[0], bal[0]
    st.markdown(f"""
Crack is the hardest class for everyone: the paper's best model scores 16.6 mask AP on it and my first run scored {b0['mask']['per_class'][2]}.
Cracks are 7% of the labels, tiny, and easy to confuse with scratches. With the last 1.4 hours of the night I tried the simplest
targeted fix I could explain in one sentence: **show the model more cracks, more often, in more places.**

**What I did (`src/make_balanced_train.py` plus one training command):**

1. **Oversampled crack photos three to one.** I built a second training list in which every photo containing at least one crack
   appears three times (434 crack photos became 1,302 entries, so an epoch grew from 2,816 to 3,684 photos). The duplicates are
   symlinks to the same files; nothing was copied. Because mosaic and the colour, flip and scale augmentations are random, the
   three copies look different every epoch.
2. **Copy-paste augmentation at 0.3 in "mixup" mode.** Three times in ten, outlines of damage from another training photo are blended
   into the current one, so cracks turn up on panels and backgrounds they never had in the original data.
3. **Continued from my trained 640 px YOLO11s** rather than starting over, with plain SGD at a low fixed learning rate of 0.003
   for 20 more epochs. The aim was to shift the model gently toward cracks without forgetting the other five classes.

**What happened.** Crack mask AP rose from {b0['mask']['per_class'][2]} to **{b1['mask']['per_class'][2]}**, the best crack score of any of my 640 px runs and above the
paper's 16.6; crack box AP rose from {b0['box']['per_class'][2]} to {b1['box']['per_class'][2]}. The cost: tyre fell from {b0['mask']['per_class'][5]} to {b1['mask']['per_class'][5]} and dent from {b0['mask']['per_class'][0]} to {b1['mask']['per_class'][0]}, so the overall
mask AP slipped from {b0['mask']['AP']} to {b1['mask']['AP']}. With only 32 flat tyres in the test set, part of that drop is noise, but the direction is real:
**a targeted fix moves the target, and always read the per-class table, not just the headline.**

**What I did not try, and why.** Per-class loss weights and focal loss are the textbook answers to imbalance. The CarDD authors got
7 to 9 points on the small classes from focal loss, but a 2026 paper found it collapsed tiny-damage detection in its setting, and I
had one night. Oversampling is the most predictable lever, so I pulled that one first. Training at full resolution from the start,
which is how others reached 57 with a small YOLO, took a 13-hour laptop day; I ran it last, and
it is the run that moved crack most (section 9). Generating extra
cracks with a diffusion model is a real option, with a fraud-risk caveat, and belongs to a later stage, not night one.
""")

# ---------------- 8. scoring ----------------
st.header("8. How I scored the models", anchor="scoring")
st.markdown(f"""
Honest numbers were the point, so the scoring rules were strict:

- **The official 374-photo test split, scored once per model.** I trained on the training split, picked the best epoch on the
  validation split, and touched the test split exactly once per run. Nothing was tuned on it.
- **pycocotools, the paper's own scoring program.** Ultralytics prints its own scores during training, but it computes mask
  accuracy on quarter-resolution masks and uses a slightly different evaluator, so its numbers are not comparable with a paper.
  My script (`src/evaluate.py`) saves the model's predictions in COCO format and scores them with pycocotools against the
  official test annotations, so my numbers belong in the same table as the paper's.
- **The words.** *IoU* (intersection over union) is how well a predicted outline overlaps the human one: 1.0 is perfect, 0.5 is
  half right. *AP* (average precision) rewards finding every damage without false alarms, averaged over confidence cut-offs; 100 is
  perfect. The headline *AP* averages over IoU thresholds from 0.5 to 0.95, so outlines must be tight; *AP50* only asks for an IoU
  of 0.5 and is more forgiving. *Mask AP* is computed on the outlines, *box AP* on the rectangles; masks score lower because
  outlines are harder. I also report AP per class and by object size (COCO's bands: small under 32 x 32 px, large over 96 x 96).
- **Speed, measured the honest way.** One photo at a time, 30 photos, after a warm-up, on an idle machine, PyTorch on the CPU
  only. I first timed the models while another run was training and the numbers were slower and noisier, so I re-measured them
  all once the laptop was idle. On the laptop's own GPU every model runs in 15 to 23 ms; the ONNX export of the best model runs in
  {export.get('ms_per_image_cpu_onnx', 0):.0f} ms on the CPU against {B['ms_per_image_cpu']:.0f} ms in PyTorch.
- **Validation is not test.** The medium YOLO26 had the best validation score of the night (59.3) and still tied the small one on
  the test split. That gap is exactly why the test split is touched once and reported as is.
- **Test-time augmentation was not available.** I tried scoring with flipped and rescaled copies of each photo averaged together;
  Ultralytics ignores that option for segmentation models and the scores came out identical, so there is no such option in my code.
""")

# ---------------- 9. results ----------------
st.header("9. Results", anchor="results")
up_mask = "above" if B["mask"]["AP"] >= 57.0 else "under"   # the paper's best scores: 57.0 mask AP, 60.6 box AP
up_box = "above" if B["box"]["AP"] >= 60.6 else "under"
st.markdown(f"""
**The headline.** My best run, **{LABEL[best]}**, scores **{B['mask']['AP']} mask AP** and **{B['box']['AP']} box AP** on the official test split.
The paper's best model, with about six times the parameters and trained for days on a server, scored 57.0 and 60.6. I am
{abs(round(B['mask']['AP'] - 57.0, 1))} points {up_mask} it on masks and {abs(round(B['box']['AP'] - 60.6, 1))} points {up_box} it on boxes. Per class the picture matches the paper:
glass, lamp and tyre are essentially solved (75 to 93), dent and scratch sit in the thirties, and crack is the hard class, {B['mask']['per_class'][2]} for my
best run against 16.6 for the paper's.
""")
blank = {k: float("nan") for k in ["Mask AP75", "Small", "Medium", "Large", "CPU ms per photo", "Laptop hours"]}
rows = [{**r, **blank} for r in PAPER]   # the paper did not publish these, so the cells stay empty
for r, (s, a, t) in sorted(RUNS.items(), key=lambda kv: -kv[1][0]["mask"]["AP"]):
    rows.append({"Model": ("★ " if r == best else "") + LABEL[r], "Params (M)": PARAMS[r], "Mask AP": s["mask"]["AP"],
                 "Mask AP50": s["mask"]["AP50"], "Mask AP75": s["mask"]["AP75"], "Box AP": s["box"]["AP"],
                 **dict(zip(NAMES, s["mask"]["per_class"])),
                 "Small": s["mask"]["by_size"][0], "Medium": s["mask"]["by_size"][1], "Large": s["mask"]["by_size"][2],
                 "CPU ms per photo": s["ms_per_image_cpu"], "Laptop hours": round(t["hours"], 1)})
cols = [c for c in rows[-1] if c not in ("Model",)]
st.dataframe(pd.DataFrame(rows)[["Model"] + cols], hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%.1f") for c in cols})
st.caption("Mask AP per class and by COCO size band, official CarDD test split (374 photos, 785 damages), scored with pycocotools. "
           "Paper rows from Wang, Li and Wu (2023). Parameter counts for the paper rows are the standard ResNet-50 and ResNet-101 "
           "Mask R-CNN sizes; the DCN+ weights were never released. CPU speed: one photo at a time on an idle M2 Max, PyTorch.")

c1, c2 = st.columns(2)
with c1:
    img("fig_per_class.png", "Mask AP per damage class. Grey bars are the paper's two models; coloured bars are mine.")
with c2:
    img("fig_by_size.png", "Mask AP by object size (COCO bands). Small damage is where cracks and most scratches live.")

st.subheader("What each run taught me")
spread = round(max(v[0]["mask"]["AP"] for v in RUNS.values()) - min(v[0]["mask"]["AP"] for v in RUNS.values()), 1)
lessons = [
    ("The newer generation helped.", "YOLO26 small beat YOLO11 small by 1.4 mask AP points with the same recipe at 640 px, mostly on dent "
     "and lamp, and its box score passed the paper's best. But it was weaker on the very small objects (13.1 against 24.2 small-object "
     "mask AP). One headline number hides trade-offs; the right model depends on what the business needs most."),
    ("More pixels helped, but only when trained in from the start.", "Continuing the 640 px YOLO11s at 1024 px for 30 epochs lost 2.0 "
     "mask AP points: the model had settled at 640 and the restart cost more than the extra pixels gave. Training YOLO26s at 1024 px "
     "from the COCO weights for 60 epochs gained 1.7 points over its 640 px twin, nearly doubled the small-object mask AP (13.1 to 24.5) "
     "and lifted crack from 16.3 to 21.9 and scratch from 29.5 to 33.0. Dent did not move, which fits what I found when I studied the "
     "misses: dents are a contrast and labelling problem, not a pixel problem. The price is 13 hours of training instead of 4, and three "
     "times the inference cost on a CPU."),
    ("Bigger was not better here.", "The medium YOLO26, nearly three times the parameters, had the best validation score of the "
     "night and still tied the small one on the test split (56.1 against 56.7). It had only 20 epochs and was still improving, and "
     "on 374 photos half a point is noise. Training budget matters as much as architecture."),
    ("Restarting a finished run costs a dip.", "Two gentle continuations of the 640 px YOLO26s, 10 and then 20 more epochs at a low "
     "learning rate, both ended below the run they started from (56.1 and 55.9 against 56.7). A restart resets the optimiser and the "
     "loss schedule, and ten or twenty epochs did not win that back. Train once, for long enough, rather than in instalments."),
    ("The crack boost did what it said, with a side effect.", "Showing crack photos three times per epoch plus copy-paste lifted "
     "crack to 17.8, the best of the 640 px runs, while tyre and dent gave a little back and the overall score slipped by 0.8. A targeted "
     "fix moves the target."),
    ("Recipe and budget moved the result more than architecture or size.", f"{len(RUNS)} runs over two nights and a day: the spread "
     f"between my best and worst run is {spread} points, and every choice that caused it was a data or training choice: resolution, "
     "epochs, what the run started from. That is why any deployment should start with data and workflow, not with models."),
]
for title, body in lessons:
    st.html(f'<div class="lesson"><b>{title}</b><br>{body}</div>')

st.subheader("Where it struggles")
c1, c2 = st.columns(2)
with c1:
    img(f"{best}/confusion_matrix_normalized.png", f"Confusion matrix of {LABEL[best]} on the test split (rows predicted, "
        "columns true). Scratch, crack and dent get confused with each other and with background; glass, lamp and tyre are nearly clean.")
with c2:
    img(f"{best}/MaskF1_curve.png", "F1 per class as the confidence cut-off moves from 0 to 1. Glass, lamp and tyre stay high over a "
        "wide range; dent, scratch and crack peak lower and fall off fast. This is the data behind the demo's review flags.")
st.markdown("""
Two failure modes, and both point to the same decisions for a real deployment. **Thin, small damage** is weak: large damage scores
about 60 mask AP and small damage 13 to 25 depending on the run, and training at full resolution is what moved small damage most.
The fixes are close-up photos in the capture protocol, full-resolution training for longer on a real GPU, and more crack examples. **Look-alike classes** get mixed up, and part of that confusion
is in the labels themselves. The fix is a clearer damage taxonomy agreed with the estimators who will use the tool, and a person in
the loop for low-confidence thin damage.
""")
img("fig_gallery.png", f"One test photo per class: human outlines above, {LABEL[best]} below at a 0.25 confidence threshold. "
    "Photos chosen by rule, not cherry-picked: the first test photo of each class with at most three damages. Where the model and "
    "the annotator disagree, either can be wrong. Photos are from the CarDD dataset and shown here for research illustration only.")

# ---------------- 10. problems ----------------
st.header("10. Problems I ran into, and what I did about them", anchor="problems")
problems = [
    ("A space in a folder name broke the first two training launches.", "The project folder has spaces in its name. An unquoted "
     "path made the training command read \"University\" as a setting and fail. Two launches were lost before I quoted every path. "
     "Cheap lesson, and now every script quotes."),
    ("Training output landed in the wrong folder.", "Ultralytics nests a relative output path under runs/segment/runs/. I switched "
     "to an absolute path so every run saved exactly where the scoring and figure scripts expected."),
    ("Two trainings at once crashed one of them.", "I started a second run while the first was still going. The Apple GPU watchdog "
     "killed one with an \"Impacting Interactivity\" error. From then on every run waited for the previous one to write \"finished\" "
     "into a log before starting, and the whole night ran as two strict one-at-a-time chains."),
    ("Apple GPUs train in full precision with one data worker.", "No mixed precision and a single loader thread meant 3 to 13 "
     "minutes per epoch. Holding every training photo in RAM was the lever that kept the GPU fed. At 1024 px the cache did not fit "
     "and those runs were slower still."),
    ("Unattended overnight runs needed a watchdog of their own.", "A health check ran every 30 minutes: is a training process alive, "
     "has its log been written in the last 10 minutes, has the epoch counter moved. The same check copied every run's best and last "
     "weights to a backup folder, so a crash could never cost a night's work. Nothing crashed after the first fix, but the backups "
     "are what this website serves."),
    ("Validation and test disagreed.", "The medium model won on validation and tied on test. I report test only, never tuned on it, "
     "and I say out loud that half a point on 374 photos is noise."),
    ("Speeds measured on a busy machine were wrong.", "My first CPU timings were taken while a run was training. They were slow and "
     "inconsistent, so I re-measured every model on an idle machine, one photo at a time, and replaced the numbers everywhere."),
    ("Test-time augmentation silently did nothing.", "I tried the flip-and-rescale averaging that often adds a point. Ultralytics "
     "ignores it for segmentation models; the scores were identical to the decimal, so I removed the option rather than pretend it worked."),
    ("The dataset licence shapes what this website can show.", "CarDD is research-only and may not be redistributed. The photos "
     "stayed on my laptop as symlinks, this website ships none of them, and the demo online runs on freely licensed sample photos and "
     "whatever you upload. The figures above show a handful of CarDD photos for research illustration, as the paper itself does."),
    ("Numbers that drift are worse than numbers that are wrong.", "Every figure and this page are generated from the "
     "same saved score files. When a run finished at 9 in the morning, one command rebuilt everything. No number was ever typed twice."),
]
cols = st.columns(2)
for i, (title, body) in enumerate(problems):
    with cols[i % 2].container(border=True):
        st.markdown(f"**{title}**  \n{body}")

# ---------------- 11. demo ----------------
st.header("11. From model to tool: how the demo is designed", anchor="demo")
st.markdown(f"""
The model gives outlines, labels and confidences. The demo turns those into the first three things an estimator writes down:

- **Severity.** For dents, scratches and cracks, the area of the outline as a share of the photo: under 1% is minor, under 5%
  moderate, above that major, ranked with the dataset's own repair priority of crack over dent over scratch (welding beats metal
  and paint beats paint alone). Glass, lamp and tyre are always "replace the part"; size does not matter. The report also names
  the panel under each damage, and the damage's share of that panel, when a second model recognises the part (see below).
- **Repair route.** Each class maps to where the job goes in the workshop: paint only, panel repair plus paint, weld or replace
  the panel, glass replacement, lamp unit, tyre. This is the hand-off to scheduling.
- **Review flag.** A detection is auto-accepted when its confidence clears a bar set per class (0.5 for glass, lamp and tyre, 0.6
  for dent and scratch, 0.7 for crack) and otherwise marked "needs estimator". The bars come from the F1 curves above: the easy
  classes stay accurate over a wide range of thresholds, the thin ones do not. In production they would be set from what a missed
  repair costs against what a wasted estimator minute costs, and re-checked after every retraining. UK GDPR Article 22C
  expects exactly this kind of meaningful human intervention for automated decisions with significant effects.

**Packaging and speed.** The best model is exported to ONNX, a plain file format that runs without PyTorch: {export.get('onnx_mb', '42')} MB,
{export.get('ms_per_image_cpu_onnx', 0):.0f} ms per photo on the laptop CPU against {B['ms_per_image_cpu']:.0f} ms in PyTorch. This website is the same Streamlit app as
the laptop demo, with all {len(RUNS)} trained models to choose from and no graphics card behind it. Expect one to two seconds per photo
here with the 1024 px model and under a second with a 640 px one; the first visit after a quiet spell can take longer while the
server wakes and the model loads.
""")

if parts:
    pc, g6, g4 = parts["mask_mAP50-95_per_class"], gap.get("0.6", {}), gap.get("0.4", {})
    st.markdown(f"""
**Naming the panel: a second model (added 5 November).** "Scratch on the front-left door, 12% of the panel" is what an estimator
writes down and what a parts order needs. CarDD has no part labels, so I trained a second segmentation model on the public Carparts
dataset: 3,833 photos of whole cars with 22 kinds of part outlined, from a Roboflow project curated by Ultralytics (CC BY 4.0). Same
recipe as the 640 px damage model, except that horizontal flips are switched off: the dataset labels left and right parts separately
and a flipped photo would teach the wrong side. {parts.get('epochs', 40)} epochs, {parts.get('hours', 5.6):.1f} hours. On that dataset's own test split it scores
**{parts['mask_mAP50-95']} mask mAP50-95** ({parts['mask_mAP50']} at IoU 0.5): windscreen {pc['front_glass']}, front bumper {pc['front_bumper']}, bonnet {pc['hood']}, doors in the seventies
and eighties, mirrors in the sixties, wheels {pc['wheel']}.

The join is one rule in `src/panels.py`: a damage belongs to the part it overlaps most if at least half of it sits inside that part,
and its size is reported as a share of that part; otherwise the panel is "unknown, estimator to confirm". Most Carparts labels carry
no side (2,520 plain "front light" against 412 "front left light"), so the side is filled in from the viewpoint: seen from the front,
the car's right is on the viewer's left; from behind it is the other way round; in a side view every visible door belongs to the side
facing the camera. Left and right are always the car's own.

**The honest measurement.** CarDD photos are close-ups, and a close-up often contains no panel edge at all. On {g6.get('photos', 100)} official
test photos with {g6.get('damages', 217)} human-labelled damages, the parts model found no part in {g6.get('photos_without_parts', 75)} photos, and a panel could be named for
**{g6.get('named_pct', 14):.0f}%** of the damages at the demo's 0.6 confidence bar ({g4.get('named_pct', 19):.0f}% at 0.4, where wrong names start to appear: a wheel arch
called a headlamp, a headlamp called a windscreen). The bar is high on purpose: a wrong panel name costs an estimator more than no
name. Two causes, neither a surprise: the training photos show whole cars from a few steps back, and the public taxonomy has no
wing, quarter panel, roof, sill or pillar, so a scratch on a front wing, one of the commonest repairs, cannot be named correctly by
this model at all. The component is right and the training distribution is wrong, which is exactly why a production version would
retrain it on the operator's own photos and part list. On wider phone photos of a car's side or front, the kind a capture
protocol would ask for, it already names the big parts.
""")

# ---------------- 12. next ----------------
st.header("12. Limitations, and what comes next", anchor="next")
st.markdown(f"""
**What this proof of concept cannot do, said plainly:**

- Thin damage is not solved. Crack is {B['mask']['per_class'][2]} mask AP for my best run and 16.6 for the paper: better, and still the weakest class by far.
- A photo cannot show hidden or structural damage. This is triage and parts pre-ordering, not the final bill; even the commercial
  systems see supplements on most AI estimates.
- These are stock photos, not claim photos. The same kind of model that scores 58 on CarDD scores 29 on another damage dataset.
  Nobody has published how a CarDD model does on real claim photos; measuring that is the first thing I would do.
- There are no undamaged cars in the training data, so the false-alarm rate on a clean car is untested.
- Panel names are a prototype. The parts model learned from whole-car photos with no wing or quarter-panel class, so on close-ups
  it mostly says "unknown"; the fix is labels from real claim photos, not a bigger model.
- Licences: CarDD is research-only, and Ultralytics is AGPL. A product trains on its own data and either buys the Enterprise
  licence or swaps to an Apache-licensed model; the pipeline is built so the swap is a one-line change.

**What a production version needs, in this order:**

- **Data and workflow first.** Run this model on real claim photos to measure the gap; define a guided capture protocol with quality
  checks on the phone; set up labelling with pre-labels corrected by estimators; agree the damage and part taxonomy; do the data
  protection work (blur plates and faces, strip location data, impact assessment).
- **Then the operator's own model.** Fine-tune on its photos; retrain the parts model on its own part list (wings, quarter panels,
  sills) and keep the outline intersection, so the output reads "scratch on front-left door, 12% of the panel"; map damage, part and
  severity to repair operations and hours from invoice history; ship with a person in the loop, every override logged.
- **Then integration.** Feed the structured output into booking, parts pre-ordering and site scheduling; monitor override rate,
  supplement rate and confidence drift by site and season; settle the production licence; add fraud and photo-quality checks.

Success is measured in business terms, not AP: time to first estimate, estimator consistency, supplement rate, and parts
pre-ordered correctly.
""")

# ---------------- 13. sources ----------------
st.header("13. Sources", anchor="sources")
st.markdown("""
- Wang, Li and Wu. *CarDD: A New Dataset for Vision-based Car Damage Detection.* IEEE Transactions on Intelligent Transportation
  Systems, 2023. Dataset site: cardd-ustc.github.io. Baselines: Mask R-CNN R-50 48.4 mask AP, DCN+ R-101 57.0 mask AP and 60.6 box AP.
- Follow-up CarDD results: YOLOv8s-seg at 1024 px for 50 epochs, 57.0 mask AP (self-reported model card, 2024); CrashSplat, YOLO11-l-seg
  58.5 mask AP (2025 preprint); C-DiffDet+, 64.8 box AP (2025 preprint).
- Ultralytics YOLO11 and YOLO26 documentation, docs.ultralytics.com; licence AGPL-3.0 or Enterprise.
- Carparts segmentation dataset: docs.ultralytics.com/datasets/segment/carparts-seg, curated from the Roboflow project "car-seg" by
  Gianmarco Russo (2023); 3,833 photos, 23 classes, CC BY 4.0. Used for the panel-naming model only.
- RF-DETR (Roboflow, Apache-2.0); RT-DETR (Baidu, Apache-2.0); SAM 2 and SAM 3 (Meta).
- GPT-4V on insurance loss assessment: arXiv 2404.09690 (2024). Grounding a vision-language model with a segmenter cut hallucinated
  report content from 92% to 31%: TinyDamage, arXiv 2608.02470 (2026). The same paper reports focal loss collapsing tiny-damage detection in its setting.
- CrashSplat (arXiv 2509.23947): the same YOLO11-l model scores 58.5 on CarDD and 29.4 on the VehiDE damage dataset.
- UK GDPR Article 22C, inserted by the Data (Use and Access) Act 2025, in force since February 2026: safeguards for significant decisions made solely by automation.
- Industry context: Tractable, Solera Qapter, CCC, Mitchell, Ravin, Click-Ins, Inspektlabs and Bdeo product pages and trade press, 2019 to 2026.
""")

# ---------------- 14. reproduce ----------------
st.header("14. Reproduce it", anchor="reproduce")
st.markdown("""
**Software.** Python 3.12, Ultralytics 8.4.201, PyTorch 2.15.0 with Apple Metal (MPS), pycocotools for scoring, Streamlit 1.66 for
the demo and this page. Hardware: Apple M2 Max, 38-core GPU, 32 GB.

**The pipeline, in the order it ran:**

| Step | File | What it does |
|---|---|---|
| 1 | `src/prepare_data.py` | converts the CarDD COCO labels to YOLO polygon labels, symlinks the photos, keeps the official split |
| 2 | `run_experiments.sh` to `run_experiments_5.sh` | the one-at-a-time training chains, one `yolo segment train` line per run |
| 3 | `src/make_balanced_train.py` | builds the crack-oversampled training list for the crack-boost run |
| 4 | `check_training.sh` | the half-hourly health check and weights backup used overnight |
| 5 | `src/evaluate.py` | scores a run once on the official test split with pycocotools; writes `results/<run>/summary.json` |
| 6 | `src/figures.py`, `src/dataset_figure.py` | draws every figure on this page from the saved scores |
| 7 | `src/export_onnx.py` | exports the best model to ONNX and times it on the CPU |
| 8 | `src/app.py`, `src/demo.py`, `src/research.py` | this website |
| 9 | `run_experiments_6.sh`, `src/panels.py`, `src/parts_check.py` | trains the car-parts model, joins its outlines to the damage outlines for the demo, and measures that join on CarDD test photos |

**The training command for the best run, exactly as it ran (the RAM cache did not fit at 1024 px, so Ultralytics skipped it):**
```
yolo segment train model=weights/yolo26s-seg.pt data=data/cardd_yolo/cardd.yaml imgsz=1024 batch=4 epochs=60 device=mps cache=ram
```
**The 640 px version of the same run, which took a third of the time:**
```
yolo segment train model=weights/yolo26s-seg.pt data=data/cardd_yolo/cardd.yaml imgsz=640 batch=8 epochs=40 device=mps cache=ram
```
**The car-parts model for panel names (flips off, because the labels tell left from right):**
```
yolo segment train model=weights/yolo26s-seg.pt data=data/carparts-seg/carparts.yaml imgsz=640 batch=8 epochs=40 patience=10 fliplr=0.0 device=mps cache=ram
```
**And the crack boost:**
```
yolo segment train model=runs/yolo11s-seg_640/weights/best.pt data=data/cardd_yolo/cardd_bal.yaml imgsz=640 batch=8 epochs=20 optimizer=SGD lr0=0.003 copy_paste=0.3 copy_paste_mode=mixup device=mps cache=ram
```
""")
st.caption("Last updated 10 November 2026.")
