"""Name the car panel under each damage, with a second model that outlines car parts.

How it works, step by step:
1. The parts model (YOLO26s-seg trained on the Carparts dataset) outlines the parts it can see: bonnet, doors, lamps...
2. That dataset names some parts with a side (front_left_door) and most without (front_door). Where the side is missing
   it is filled in from the viewpoint: seen from the front, the car's right is on the viewer's left; seen from behind it
   is the other way round; in a side view every visible door, mirror and wheel belongs to the side facing the camera.
3. For each damage outline, find the part it overlaps most. If at least half of the damage sits inside that part, the
   damage is on it and its size is reported as a share of the part. Otherwise the panel is unknown and an estimator
   should confirm.
Left and right always mean the car's own left and right (nearside and offside in UK bodyshop language).
Run this file directly for a small self-check with made-up outlines; no model is needed for that.
"""
import cv2, numpy as np

# Carparts class name -> (plain part name, side). The side is the car's own left or right, or None when the label did not say.
PART = {"back_bumper": ("rear bumper", None), "back_door": ("rear door", None), "back_glass": ("rear window", None),
        "back_left_door": ("rear door", "left"), "back_left_light": ("rear light", "left"), "back_light": ("rear light", None),
        "back_right_door": ("rear door", "right"), "back_right_light": ("rear light", "right"),
        "front_bumper": ("front bumper", None), "front_door": ("front door", None), "front_glass": ("windscreen", None),
        "front_left_door": ("front door", "left"), "front_left_light": ("headlamp", "left"), "front_light": ("headlamp", None),
        "front_right_door": ("front door", "right"), "front_right_light": ("headlamp", "right"), "hood": ("bonnet", None),
        "left_mirror": ("mirror", "left"), "object": ("other", None), "right_mirror": ("mirror", "right"),
        "tailgate": ("tailgate", None), "trunk": ("boot lid", None), "wheel": ("wheel", None)}
SIDED = {"front door", "rear door", "headlamp", "rear light", "mirror", "wheel"}     # parts that exist once per side
PAIRED = {"headlamp", "rear light", "mirror"}                                        # sided parts that come in a visible pair
FRONT = {"windscreen", "bonnet", "front bumper", "headlamp"}                         # the front end of the car
BEHIND = {"rear window", "boot lid", "tailgate", "rear bumper", "rear light", "front door", "rear door", "mirror"}   # everything behind it


def detect_parts(model, bgr, device="cpu", conf=0.6):
    """Run the parts model on one photo. Returns one dict per part: name, side, conf, poly (pixel outline).
    The bar is high on purpose (0.6): a wrong panel name costs an estimator more than no name."""
    res = model.predict(bgr, imgsz=640, conf=conf, device=device, verbose=False)[0]
    parts = []
    if res.masks is not None:
        for poly, cls, c in zip(res.masks.xy, res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            name, side = PART[model.names[int(cls)]]
            if name != "other" and len(poly) >= 3:
                parts.append({"name": name, "side": side, "conf": float(c), "poly": poly.astype(np.int32)})
    h, w = bgr.shape[:2]
    parts = _dedup(parts, w, h)
    _fill_sides(parts)
    return parts


def name_panels(damage_polys, parts, w, h, min_inside=0.5):
    """For each damage outline: (panel label, damage size as % of that panel, index of the part), or ("unknown", None, None)."""
    pm, dm = _rasterise([p["poly"] for p in parts], w, h), _rasterise(damage_polys, w, h)
    out = []
    for d in dm:
        inside = [np.logical_and(d, p).sum() for p in pm]
        if not inside or d.sum() == 0 or max(inside) / d.sum() < min_inside:
            out.append(("unknown", None, None)); continue
        k = int(np.argmax(inside)); p = parts[k]
        label = p["name"] + (f", {p['side']}" if p["side"] else ", side unknown" if p["name"] in SIDED else "")
        out.append((label, round(100 * inside[k] / pm[k].sum(), 1), k))
    return out


def _dedup(parts, w, h):
    """The model often outlines one lamp twice, as 'front_light' and as 'front_left_light'. Keep one: sided first, then the surer."""
    masks = _rasterise([p["poly"] for p in parts], w, h)
    kept = []
    for i in sorted(range(len(parts)), key=lambda i: (parts[i]["side"] is not None, parts[i]["conf"]), reverse=True):
        if not any(parts[j]["name"] == parts[i]["name"] and _iou(masks[i], masks[j]) > 0.6 for j in kept):
            kept.append(i)
    return [parts[i] for i in kept]


def _fill_sides(parts):
    """Fill in missing sides from the viewpoint (changes the dicts in place).
    ponytail: a heuristic. Labels with sides on real claim photos make it unnecessary."""
    cx = lambda p: float(p["poly"][:, 0].mean())
    front = [cx(p) for p in parts if p["name"] in FRONT]
    behind = [cx(p) for p in parts if p["name"] in BEHIND]
    # which side of the car faces the camera: front end on the viewer's left means we are looking at the car's left side
    side_seen = ("left" if np.mean(front) < np.mean(behind) else "right") if front and behind else None
    rear_view = bool(behind) and not front
    for name in SIDED:
        group = [p for p in parts if p["name"] == name and p["side"] is None]
        if len(group) >= 2 and name in PAIRED:
            group.sort(key=cx)      # from the front the car's right is on the viewer's left; from behind it is the car's left
            group[0]["side"], group[-1]["side"] = ("left", "right") if rear_view else ("right", "left")
        elif side_seen:
            for p in group:
                p["side"] = side_seen


def _rasterise(polys, w, h, long_side=400):
    """Draw each outline as a small true/false picture, so overlaps are simple pixel counts."""
    s = long_side / max(w, h)
    H, W = max(1, round(h * s)), max(1, round(w * s))
    masks = []
    for poly in polys:
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.round(np.asarray(poly, float) * s).astype(np.int32)], 1)
        masks.append(m.astype(bool))
    return masks


def _iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 0.0


if __name__ == "__main__":
    sq = lambda x, y, s: np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]])
    W, H = 800, 600
    # a three-quarter view: bonnet and two headlamps on the right, a door on the left -> the car's right side faces us
    parts = [{"name": "front door", "side": None, "conf": .9, "poly": sq(50, 100, 200)},
             {"name": "headlamp", "side": None, "conf": .9, "poly": sq(400, 350, 50)},
             {"name": "headlamp", "side": None, "conf": .8, "poly": sq(700, 350, 50)},
             {"name": "headlamp", "side": None, "conf": .7, "poly": sq(705, 352, 48)},      # duplicate outline of the same lamp
             {"name": "bonnet", "side": None, "conf": .9, "poly": sq(450, 350, 250)}]
    parts = _dedup(parts, W, H)
    assert [p["name"] for p in parts].count("headlamp") == 2, "duplicate lamp should be dropped"
    _fill_sides(parts)
    sides = {(p["name"], int(p["poly"][0, 0])): p["side"] for p in parts}
    assert sides[("front door", 50)] == "right", sides
    assert sides[("headlamp", 400)] == "right" and sides[("headlamp", 700)] == "left", sides   # viewer-left lamp is the car's right one
    assert all(p["side"] is None for p in parts if p["name"] == "bonnet")
    got = name_panels([sq(100, 150, 50), sq(500, 0, 50), sq(240, 150, 40)], parts, W, H)
    assert got[0][0] == "front door, right" and abs(got[0][1] - 6.2) < 0.5, got[0]   # 50x50 inside a 200x200 door = 6.25%
    assert got[1] == ("unknown", None, None), got[1]                                   # nowhere near a part
    assert got[2][0] == "unknown", got[2]                                              # only a quarter of it sits on the door
    # a plain rear view: two rear lights, nothing from the front -> the viewer's left lamp is the car's left one
    rear = [{"name": "rear light", "side": None, "conf": .9, "poly": sq(100, 300, 60)},
            {"name": "rear light", "side": None, "conf": .9, "poly": sq(600, 300, 60)},
            {"name": "rear bumper", "side": None, "conf": .9, "poly": sq(100, 400, 560)}]
    _fill_sides(rear)
    assert rear[0]["side"] == "left" and rear[1]["side"] == "right"
    print("panels.py self-check passed")
