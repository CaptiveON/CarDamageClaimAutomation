#!/bin/bash
# Fifth chain, started 4 Nov 22:30 on the user's go. One job only:
#   E8: YOLO26s at 1024 px, started fresh from the COCO weights, 60 epochs, batch 4.
#   Every other setting is the same as the 640 run, so the only change is the picture size.
# The half-hourly check may stop the run after epoch 50 if the validation score has stopped improving.
# Either way this script then backs up best.pt and scores it on the official test split.
cd "$(dirname "$0")"
Y=.venv/bin/yolo; P=.venv/bin/python; OUT="$PWD/runs"
backup_and_score () {
  mkdir -p "weights_trained/$1"
  cp "runs/$1/weights/best.pt" "runs/$1/weights/last.pt" "runs/$1/args.yaml" "runs/$1/results.csv" "weights_trained/$1/" 2>/dev/null
  $P src/evaluate.py "runs/$1" mps > "results/eval_$1.log" 2>&1
  echo "$1 scored $(date): $($P -c "import json; s=json.load(open('results/$1/summary.json')); print('mask', s['mask']['AP'], 'box', s['box']['AP'], 'crack', s['mask']['per_class'][2])" 2>&1)" >> runs/driver.log
}
while pgrep -f "yolo segment train|src/evaluate.py" >/dev/null; do sleep 60; done   # never two jobs on the GPU
echo "E8 starts (1024 px, 60 epochs, from COCO weights) $(date)" >> runs/driver.log
$Y segment train model=weights/yolo26s-seg.pt data=data/cardd_yolo/cardd.yaml imgsz=1024 batch=4 epochs=60 device=mps cache=ram "project=$OUT" name=yolo26s-seg_1024 exist_ok=True > runs/e8.log 2>&1
echo "E8 finished $(date)" >> runs/driver.log
backup_and_score yolo26s-seg_1024
echo "E8 scored; chain 5 done $(date)" >> runs/driver.log
