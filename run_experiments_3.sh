#!/bin/bash
# Third chain (4 Nov evening). Waits for E6 to finish AND be scored, then runs one job at a time:
#   E6b: 10 more gentle epochs on the E6 model (added 17:20 at the user's request).
#   E7: the best model (YOLO26s, 640 px) given the crack boost: crack photos shown three times per epoch + copy-paste,
#       SGD at a low fixed learning rate, 20 epochs. Same recipe as the YOLO11s crack boost, so the two are comparable.
#   E8: YOLO26s trained at 1024 px FROM the COCO weights for 40 epochs (the recipe others used to reach 57 with a small
#       model; our earlier 1024 run only continued a 640 model and lost points). Batch 4 to fit the GPU memory. ~10 hours.
# After each run: copy the weights to weights_trained/<run>/ and score once on the official test split (nothing else on the GPU).
cd "$(dirname "$0")"
Y=.venv/bin/yolo; P=.venv/bin/python; OUT="$PWD/runs"

backup_and_score () {   # $1 = run name
  mkdir -p "weights_trained/$1"
  cp "runs/$1/weights/best.pt" "runs/$1/weights/last.pt" "runs/$1/args.yaml" "runs/$1/results.csv" "weights_trained/$1/" 2>/dev/null
  $P src/evaluate.py "runs/$1" mps > "results/eval_$1.log" 2>&1
  echo "$1 scored $(date): $($P -c "import json; s=json.load(open('results/$1/summary.json')); print('mask', s['mask']['AP'], 'box', s['box']['AP'], 'crack', s['mask']['per_class'][2])" 2>&1)" >> runs/driver.log
}

# wait until E6 has been scored (its own command does that); fall back to "finished and nothing running on the GPU"
until [ -f results/yolo26s-seg_640_cont/summary.json ]; do
  if grep -q "E6 finished" runs/driver.log && ! pgrep -f "yolo segment train|src/evaluate.py" >/dev/null; then sleep 180; break; fi
  sleep 60
done

# E6b: ten more gentle epochs on top of E6 (user asked for 10 more). Starts from E6's best weights, at a lower rate (0.0002)
#      than E6 so the restart dip is small; the learning rate then decays to 1% of that over the 10 epochs.
echo "E6b starts $(date)" >> runs/driver.log
$Y segment train model=runs/yolo26s-seg_640_cont/weights/best.pt data=data/cardd_yolo/cardd.yaml imgsz=640 batch=8 epochs=10 optimizer=AdamW lr0=0.0002 momentum=0.9 warmup_epochs=0.5 device=mps cache=ram "project=$OUT" name=yolo26s-seg_640_cont2 exist_ok=True > runs/e6b.log 2>&1
echo "E6b finished $(date)" >> runs/driver.log
backup_and_score yolo26s-seg_640_cont2

echo "E7 starts $(date)" >> runs/driver.log
$Y segment train model=weights_trained/yolo26s-seg_640/best.pt data=data/cardd_yolo/cardd_bal.yaml imgsz=640 batch=8 epochs=20 optimizer=SGD lr0=0.003 copy_paste=0.3 copy_paste_mode=mixup device=mps cache=ram "project=$OUT" name=yolo26s-seg_640_bal exist_ok=True > runs/e7.log 2>&1
echo "E7 finished $(date)" >> runs/driver.log
backup_and_score yolo26s-seg_640_bal

echo "E8 starts $(date)" >> runs/driver.log
$Y segment train model=weights/yolo26s-seg.pt data=data/cardd_yolo/cardd.yaml imgsz=1024 batch=4 epochs=40 device=mps cache=ram "project=$OUT" name=yolo26s-seg_1024 exist_ok=True > runs/e8.log 2>&1
echo "E8 finished $(date)" >> runs/driver.log
backup_and_score yolo26s-seg_1024
echo "E8 scored; chain 3 done $(date)" >> runs/driver.log
