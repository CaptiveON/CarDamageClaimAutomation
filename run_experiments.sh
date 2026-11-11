#!/bin/bash
# Runs the remaining training experiments one after another (one at a time: two at once made the Mac's GPU watchdog kill a run).
#   E1: YOLO11s-seg at 640 px, 40 epochs            done 21:05 -> runs/yolo11s-seg_640
#   E3: the E1 model trained further at 1024 px     (full resolution, where small cracks and scratches live)
#   E2: YOLO26s-seg at 640 px, 40 epochs            (newest generation, for the generation comparison)
# "yolo" is the Ultralytics command line tool. One line = one training run. Logs go to runs/*.log.
# The project folder name has spaces, so every path is wrapped in quotes.
cd "$(dirname "$0")"
YOLO=.venv/bin/yolo
DATA=data/cardd_yolo/cardd.yaml
OUT="$PWD/runs"          # absolute path, otherwise Ultralytics nests it under runs/segment/runs/

# E3: keep training the 640 px model at 1024 px. Batch 4 keeps GPU memory and kernel size modest.
"$YOLO" segment train model=runs/yolo11s-seg_640/weights/best.pt data=$DATA imgsz=1024 batch=4 epochs=30 device=mps cache=ram "project=$OUT" name=yolo11s-seg_1024 exist_ok=True > runs/e3.log 2>&1
echo "E3 finished $(date)" >> runs/driver.log

# E2: YOLO26s at 640 px from the COCO-pretrained weights, same recipe as E1
"$YOLO" segment train model=weights/yolo26s-seg.pt data=$DATA imgsz=640 batch=8 epochs=40 device=mps cache=ram "project=$OUT" name=yolo26s-seg_640 exist_ok=True > runs/e2.log 2>&1
echo "E2 finished $(date); all done" >> runs/driver.log
