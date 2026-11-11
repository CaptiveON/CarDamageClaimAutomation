#!/bin/bash
# Sixth chain, started 5 Nov on the user's go. One job, the car-parts (panel) model:
#   E9: YOLO26s-seg at 640 px from the COCO weights on the Ultralytics Carparts dataset (23 part classes, CC BY 4.0),
#       40 epochs, batch 8, early stopping with patience 10. Same recipe as the 640 px damage model, except horizontal
#       flips are off: the dataset labels left and right parts separately and a flipped photo would teach the wrong side.
# It is scored on the dataset's own test split with the Ultralytics validator, not with src/evaluate.py (that one is for CarDD).
cd "$(dirname "$0")"
Y=.venv/bin/yolo; P=.venv/bin/python; OUT="$PWD/runs"; RUN=parts_yolo26s_640
while pgrep -f "yolo segment train|src/evaluate.py" >/dev/null; do sleep 60; done
echo "E9 starts (parts model, 640 px, 40 epochs, patience 10, from COCO weights) $(date)" >> runs/driver.log
$Y segment train model=weights/yolo26s-seg.pt data=data/carparts-seg/carparts.yaml imgsz=640 batch=8 epochs=40 patience=10 fliplr=0.0 device=mps cache=ram "project=$OUT" name=$RUN exist_ok=True > runs/e9.log 2>&1
echo "E9 finished $(date)" >> runs/driver.log
mkdir -p "weights_trained/$RUN" "results/$RUN"
cp "runs/$RUN/weights/best.pt" "runs/$RUN/weights/last.pt" "runs/$RUN/args.yaml" "runs/$RUN/results.csv" "weights_trained/$RUN/" 2>/dev/null
# score once on the test split and keep the numbers as a small json for the demo and the notes
RUN=$RUN OUT=$OUT $P - <<'PY' > "results/$RUN/val_test.log" 2>&1
import json, os
from ultralytics import YOLO
run, out = os.environ["RUN"], os.environ["OUT"]
m = YOLO(f"runs/{run}/weights/best.pt")
r = m.val(data="data/carparts-seg/carparts.yaml", split="test", imgsz=640, batch=8, device="mps", plots=False,
          project=out, name=f"{run}_test", exist_ok=True, verbose=False)
res = {"run": run, "split": "test", "imgsz": 640,
       "box_mAP50-95": round(r.box.map * 100, 1), "box_mAP50": round(r.box.map50 * 100, 1),
       "mask_mAP50-95": round(r.seg.map * 100, 1), "mask_mAP50": round(r.seg.map50 * 100, 1),
       "mask_mAP50-95_per_class": {m.names[i]: round(float(v) * 100, 1) for i, v in enumerate(r.seg.maps)}}
json.dump(res, open(f"results/{run}/val_test.json", "w"), indent=2)
print(res)
PY
echo "E9 scored on the Carparts test split $(date): $($P -c "import json; d=json.load(open('results/$RUN/val_test.json')); print('mask mAP50-95', d['mask_mAP50-95'], 'box', d['box_mAP50-95'])" 2>&1)" >> runs/driver.log
echo "E9 scored; chain 6 done $(date)" >> runs/driver.log
