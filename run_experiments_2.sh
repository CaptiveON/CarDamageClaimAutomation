#!/bin/bash
# Second chain, started by itself once the YOLO26 run (E2) has finished. One run at a time, logs in runs/*.log.
#   E5: YOLO26m-seg (the medium model of the newer generation, 27 M parameters) at 640 px, 20 epochs.
#       YOLO26 small beat YOLO11 small by about two points on validation, so the medium run uses YOLO26 too.
#       20 epochs, not 30: the medium model takes about three times longer per epoch and time was short.
#   E4: the best YOLO11s model so far, trained 20 more epochs at 640 px with two changes aimed at the weak classes:
#       (a) photos that contain a crack are shown three times per epoch (data/cardd_yolo/cardd_bal.yaml)
#       (b) copy-paste augmentation: damage cut out of one photo and pasted into another during training
#       Kept at 640 px so the only difference from the 640 px model is the crack boost (the 1024 px continuation did not help).
#       A low, fixed learning rate because this is a gentle fine-tune, not training from scratch.
cd "$(dirname "$0")"
until grep -q "E2 finished" runs/driver.log; do sleep 60; done

echo "E5 starts $(date)" >> runs/driver.log
.venv/bin/yolo segment train model=weights/yolo26m-seg.pt data=data/cardd_yolo/cardd.yaml imgsz=640 batch=8 epochs=20 device=mps cache=ram "project=$PWD/runs" name=yolo26m-seg_640 exist_ok=True > runs/e5.log 2>&1
echo "E5 finished $(date)" >> runs/driver.log

BEST=$(.venv/bin/python -c "import json,glob; r={f.split('/')[1]: json.load(open(f))['mask']['AP'] for f in glob.glob('results/yolo11s-seg_*/summary.json')}; print(max(r, key=r.get))")
echo "E4 starts from $BEST $(date)" >> runs/driver.log
.venv/bin/yolo segment train model=runs/$BEST/weights/best.pt data=data/cardd_yolo/cardd_bal.yaml imgsz=640 batch=8 epochs=20 optimizer=SGD lr0=0.003 copy_paste=0.3 copy_paste_mode=mixup device=mps cache=ram "project=$PWD/runs" name=yolo11s-seg_640_bal exist_ok=True > runs/e4.log 2>&1
echo "E4 finished $(date); chain 2 done" >> runs/driver.log
