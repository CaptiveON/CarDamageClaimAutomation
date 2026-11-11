"""Live demo page: upload a car photo (or try a sample), see the damage outlines and a plain damage report.
On my laptop the official CarDD test photos and their human annotations are available; on the website they are not
(the dataset is research-only), so the page falls back to freely licensed sample photos and uploads.
"""
import json, random, time
from pathlib import Path
import numpy as np, streamlit as st
# The heavy libraries load inside a spinner: on the web server the first visit after a quiet spell starts a fresh
# machine, and importing PyTorch there takes 10 to 20 seconds. Later imports are instant, so the spinner only flashes.
with st.spinner("The server is warming up. The first visit after a quiet spell takes up to a minute; after that "
                "each photo takes a second or two.", show_time=True):
    import cv2, torch
    from ultralytics import YOLO
    from ultralytics.utils import YAML
    from panels import detect_parts, name_panels   # the car-parts model that names the panel under each damage

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "data/cardd_yolo/images/test"                              # official test photos (laptop only)
GT_FILE = ROOT / "Dataset/CarDD_COCO/annotations/instances_test2017.json"   # their human annotations (laptop only)
SAMPLES = ROOT / "samples"                                                   # freely licensed sample photos (website)
PARTS = ROOT / "weights_trained/parts_yolo26s_640/best.pt"                    # the car-parts model for panel names (optional)
# The first photo a visitor sees, chosen by hand so the first impression shows several kinds of damage found with high
# confidence: on the laptop a CarDD test photo (dent, broken lamp and scratch, all above 0.85), on the website the
# shattered-windscreen sample (glass 0.93 plus a dented bonnet). The button then picks a different photo at random.
SHOWCASE = TEST_DIR / "003441.jpg" if TEST_DIR.exists() else SAMPLES / "glass.jpg"
NAMES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
COLORS = [(255, 56, 56), (255, 200, 0), (0, 220, 255), (255, 0, 255), (0, 255, 80), (255, 140, 0)]  # one colour per class
# Priority copied from the dataset's own labelling rule: crack > dent > scratch (welding > metal + paint > paint).
# ponytail: real severity needs the car part and repair-time data; this is a readable placeholder for the demo.
PRIORITY = {"crack": 3, "dent": 2, "scratch": 1}
# Where each damage type goes in the workshop. This is the hand-off to scheduling: paint bay, panel bay, glass, parts.
REPAIR = {"scratch": "paint only", "dent": "panel repair + paint", "crack": "weld or replace panel",
          "glass shatter": "glass replacement", "lamp broken": "replace lamp unit", "tire flat": "tyre replacement"}
# Minimum confidence to accept a detection without a person looking at it. Thin damage needs a higher bar.
# ponytail: placeholder values; in production these come from the per-class F1 curves and the cost of a miss vs a false alarm.
AUTO_ACCEPT = {"glass shatter": 0.5, "lamp broken": 0.5, "tire flat": 0.5, "dent": 0.6, "scratch": 0.6, "crack": 0.7}


@st.cache_resource
def load_model(path):
    return YOLO(path)


@st.cache_data
def ground_truth():
    """file name -> list of (polygon, class index) from the official test annotations. Empty when the dataset is not here."""
    if not GT_FILE.exists():
        return {}
    d = json.load(open(GT_FILE))
    name_of = {i["id"]: i["file_name"] for i in d["images"]}
    gt = {}
    for a in d["annotations"]:
        poly = np.array(a["segmentation"][0], dtype=np.int32).reshape(-1, 2)
        gt.setdefault(name_of[a["image_id"]], []).append((poly, a["category_id"] - 1))
    return gt


def draw(img, items, parts=()):
    """Paint each (polygon, class, text) on a copy of the image: filled colour at 40% plus an outline and a label.
    Car parts, when given, go underneath as thin white outlines with a small name."""
    out = img.copy(); overlay = img.copy()
    for poly, cls, text in items:
        cv2.fillPoly(overlay, [poly], COLORS[cls])
    out = cv2.addWeighted(overlay, 0.4, out, 0.6, 0)
    for p in parts:
        cv2.polylines(out, [p["poly"]], True, (245, 245, 245), 1)
        x, y = p["poly"].mean(0).astype(int)
        cv2.putText(out, p["name"] + (f" ({p['side']})" if p["side"] else ""), (int(x) - 30, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1)
    for poly, cls, text in items:
        cv2.polylines(out, [poly], True, COLORS[cls], 2)
        x, y = poly[0]
        cv2.putText(out, text, (int(x), max(int(y) - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[cls], 2)
    return out


def severity(name, area_pct):
    if name not in PRIORITY:
        return "replace part"                      # glass, lamp, tyre: you swap the part, size does not matter
    return "minor" if area_pct < 1 else "moderate" if area_pct < 5 else "major"


def test_ap(p):
    """Mask AP of this run on the official test split, or -1 if it was never scored."""
    f = ROOT / "results" / p.parent.name / "summary.json"
    return json.load(open(f))["mask"]["AP"] if f.exists() else -1


# ---------------- page ----------------
st.title("Car damage detection and segmentation")
st.markdown(
    "Upload a photo of a damaged car. The model outlines every damage it finds, names it and gives a confidence. "
    "The report underneath turns that into a severity, a repair route and a review flag, the way an estimator's "
    "first pass would. I trained these models on the public CarDD dataset over two nights and a day on a laptop; "
    "the **Method and results** page in the sidebar explains how and what I learned.")
if not TEST_DIR.exists():
    st.caption("This runs on a small shared server with no graphics card that goes to sleep when nobody is using it. "
               "The first visit after a quiet spell can take up to a minute to wake it; then each photo takes a second or two.")

runs = sorted([p for p in ROOT.glob("weights_trained/*/best.pt") if not p.parent.name.startswith("parts_")],
              key=test_ap, reverse=True)   # best scored damage run first; the parts model is not a damage model
runs = [p for p in runs if test_ap(p) >= 0] or runs                                  # hide unscored runs when a scored one exists
if not runs:
    st.error("No trained model found (weights_trained/*/best.pt)."); st.stop()

st.sidebar.header("Settings")
weights = st.sidebar.selectbox("Model", runs, format_func=lambda p: f"{p.parent.name}  (test mask AP {test_ap(p)})",
                               help="The score is mask AP on the official 374-photo CarDD test split; higher is better.")
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05,
                         help="Predictions below this confidence are hidden. Lower it to see more, raise it to see only confident ones.")
devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
device = st.sidebar.selectbox("Run on", devices, help="cpu is what an office PC, or this web server, has.")
imgsz = YAML.load(weights.parent / "args.yaml")["imgsz"]   # same picture size as in training
with st.spinner("Loading the model...", show_time=True):
    model = load_model(str(weights))

st.sidebar.header("Photo")
upload = st.sidebar.file_uploader("Upload a car photo", type=["jpg", "jpeg", "png"])
pool = sorted(TEST_DIR.glob("*.jpg")) if TEST_DIR.exists() else sorted(SAMPLES.glob("*.jpg"))
label = "Random test photo" if TEST_DIR.exists() else "Try a sample photo"
if pool and "pick" not in st.session_state:
    st.session_state.pick = SHOWCASE if SHOWCASE.exists() else pool[0]
if pool and st.sidebar.button(label, width="stretch"):
    st.session_state.pick = random.choice([p for p in pool if p != st.session_state.pick] or pool)
if upload:
    img = cv2.imdecode(np.frombuffer(upload.read(), np.uint8), cv2.IMREAD_COLOR); name = upload.name
elif pool:
    img = cv2.imread(str(st.session_state.pick)); name = st.session_state.pick.name
else:
    st.info("Upload a photo of a damaged car in the sidebar to start."); st.stop()
if img is None:
    st.error("That file could not be read as an image. Try a JPEG or PNG."); st.stop()
bgr = img                                     # OpenCV order, which is what the model expects
img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)    # normal colour order, for showing on screen
h, w = img.shape[:2]

with st.spinner("Looking for damage...", show_time=True):
    t = time.time()
    res = model.predict(bgr, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    parts = detect_parts(load_model(str(PARTS)), bgr, device) if PARTS.exists() else []   # the car's panels, when that model is here
    ms = (time.time() - t) * 1000

# turn the model output into rows: one per damage instance
found = []                                             # (outline, class index, confidence) of every damage kept
if res.masks is not None:
    for poly, cls, c in zip(res.masks.xy, res.boxes.cls.tolist(), res.boxes.conf.tolist()):
        if len(poly) >= 3:                             # a mask too tiny to form a shape is skipped
            found.append((poly.astype(np.int32), int(cls), c))
panels = name_panels([f[0] for f in found], parts, w, h)   # which panel each damage sits on, and how much of it
rows, items = [], []
for (poly, cls, c), (panel, share, _) in zip(found, panels):
    area_pct = cv2.contourArea(poly) / (w * h) * 100
    n = NAMES[cls]
    row = {"damage": n, "confidence": round(c, 2), "area % of image": round(area_pct, 2)}
    if PARTS.exists():
        row |= {"panel": panel, "area % of panel": share}                   # None when the panel is unknown
    row |= {"severity": severity(n, area_pct), "repair route": REPAIR[n],
            "review": "auto-accept" if c >= AUTO_ACCEPT[n] else "needs estimator"}
    rows.append(row)
    items.append((poly, cls, f"{n} {c:.2f}"))

left, right = st.columns(2)
left.subheader(f"Prediction  ({ms:.0f} ms)"); left.image(draw(img, items, parts), width="stretch")
gt = ground_truth().get(name)
if gt:
    right.subheader("Ground truth (human annotation)")
    right.image(draw(img, [(p, c, NAMES[c]) for p, c in gt]), width="stretch")
else:
    right.subheader("Original photo")
    right.image(img, width="stretch")
    right.caption("No human annotation exists for this photo, so judge the outlines yourself.")

st.subheader("Damage report")
if rows:
    shown = [r | {"area % of panel": "" if r["area % of panel"] is None else f"{r['area % of panel']:.1f} %"} if "area % of panel" in r else r
             for r in rows]                                   # an unknown panel shows a blank cell, not the word None
    st.dataframe(shown, hide_index=True, width="stretch",
                 column_config={"confidence": st.column_config.NumberColumn(format="%.2f"),
                                "area % of image": st.column_config.NumberColumn(format="%.2f %%")})
    worst = max(rows, key=lambda r: (PRIORITY.get(r["damage"], 4), r["area % of image"]))
    n_review = sum(r["review"] == "needs estimator" for r in rows)
    routes = sorted({r["repair route"] for r in rows})
    with st.container(border=True):
        st.markdown(f"**Overall:** {len(rows)} damage(s) found. Most serious: **{worst['damage']}**"
                    f"{' on the ' + worst['panel'] if worst.get('panel') not in (None, 'unknown') else ''} ({worst['severity']}). "
                    f"Workshop routes: {', '.join(routes)}. "
                    + (f"**{n_review} item(s) need an estimator's look** before booking." if n_review
                       else "**All items confident enough to pre-fill the estimate.**"))
else:
    st.markdown("**No damage found above the confidence threshold.** Lower the threshold in the sidebar to see weaker predictions.")
# "Ask about this job": a laptop-only prototype that answers questions from mock records and a pricing function (rag/ask.py).
# It needs an API key in the environment, and the rag folder is never uploaded, so the website never shows it.
try:
    import os, sys
    if str(ROOT / "rag") not in sys.path:
        sys.path.append(str(ROOT / "rag"))
    from ask import render_ask
    ASK = bool(os.environ.get("ANTHROPIC_API_KEY"))
except Exception:
    ASK = False
if ASK and rows:
    render_ask(rows)
st.caption(f"Model: {weights.parent.name}, input {imgsz} px, running on {device}. "
           "Severity bands: under 1% of the photo is minor, under 5% moderate, above that major; glass, lamp and tyre are always "
           "'replace part'. Review flag: auto-accept when the confidence is at least 0.5 for glass, lamp and tyre, 0.6 for dent "
           "and scratch, 0.7 for crack. These bars are placeholders for the demo; in a real deployment they come from the cost "
           "of a missed repair versus a wasted estimator minute."
           + (" Panels come from a second model, YOLO26s trained on the public Carparts dataset, and are named only when that model "
              "is at least 60% sure and at least half of the damage sits inside the part; otherwise 'unknown'. Left and right are "
              "the car's own sides. On close-up photos the panel is often out of frame, so 'unknown' is common." if PARTS.exists() else ""))
credits = SAMPLES / "CREDITS.md"
if credits.exists() and not TEST_DIR.exists():
    with st.expander("Sample photo credits"):
        st.markdown(credits.read_text())
