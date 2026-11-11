#!/bin/bash
# Fourth chain, launched by hand after the user's review (4 Nov evening). One job at a time, each backed up and scored.
#   E7: YOLO26s (the best model) with the crack boost: crack photos x3 per epoch + copy-paste, SGD at a low fixed rate, 20 epochs.
#   E8: YOLO26s at 1024 px from the COCO weights, 40 epochs, batch 4 (the full-resolution recipe; about 10 hours).
cd "$(dirname "$0")"
Y=.venv/bin/yolo; P=.venv/bin/python; OUT="$PWD/runs"
backup_and_score () {
  mkdir -p "weights_trained/$1"
  cp "runs/$1/weights/best.pt" "runs/$1/weights/last.pt" "runs/$1/args.yaml" "runs/$1/results.csv" "weights_trained/$1/" 2>/dev/null
  $P src/evaluate.py "runs/$1" mps > "results/eval_$1.log" 2>&1
  echo "$1 scored $(date): $($P -c "import json; s=json.load(open('results/$1/summary.json')); print('mask', s['mask']['AP'], 'box', s['box']['AP'], 'crack', s['mask']['per_class'][2])" 2>&1)" >> runs/driver.log
}
while pgrep -f "yolo segment train|src/evaluate.py" >/dev/null; do sleep 60; done   # never two jobs on the GPU
echo "E7 starts $(date)" >> runs/driver.log
$Y segment train model=weights_trained/yolo26s-seg_640/best.pt data=data/cardd_yolo/cardd_bal.yaml imgsz=640 batch=8 epochs=20 optimizer=SGD lr0=0.003 copy_paste=0.3 copy_paste_mode=mixup device=mps cache=ram "project=$OUT" name=yolo26s-seg_640_bal exist_ok=True > runs/e7.log 2>&1
echo "E7 finished $(date)" >> runs/driver.log
backup_and_score yolo26s-seg_640_bal
echo "E8 starts $(date)" >> runs/driver.log
$Y segment train model=weights/yolo26s-seg.pt data=data/cardd_yolo/cardd.yaml imgsz=1024 batch=4 epochs=40 device=mps cache=ram "project=$OUT" name=yolo26s-seg_1024 exist_ok=True > runs/e8.log 2>&1
echo "E8 finished $(date)" >> runs/driver.log
backup_and_score yolo26s-seg_1024
echo "E8 scored; chain 4 done $(date)" >> runs/driver.log
