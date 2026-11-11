#!/bin/bash
# One-screen health check of the overnight training. Run any time:  ./check_training.sh
# Healthy = a yolo process exists, its log was written in the last few minutes, and the epoch counter moves.
cd "$(dirname "$0")"
date
echo "--- driver log"; cat runs/driver.log
echo "--- training processes"; pgrep -fl "yolo segment train" | cut -c1-70 || echo "none running"
for f in $(ls -t runs/e[0-9]*.log 2>/dev/null | head -3); do   # the three most recent training logs
  [ -f "$f" ] || continue
  age=$(( $(date +%s) - $(stat -f %m "$f") ))
  echo "--- $f  (last written ${age}s ago)"
  LC_ALL=C tail -c 1500 "$f" | LC_ALL=C tr '\r' '\n' | grep -E "[0-9]+/[0-9]+ +[0-9.]+G|Error|Traceback| all " | tail -2 | cut -c1-110
done
echo "--- validation scores so far (epoch: box mAP50-95 / mask mAP50-95), last 3 epochs"
.venv/bin/python - <<'PY'
import csv, pathlib
for f in sorted(pathlib.Path("runs").glob("*/results.csv"), key=lambda f: f.stat().st_mtime)[-3:]:   # the three most recent runs
    d = f.parent.name
    if True:
        rows = list(csv.DictReader(open(f)))
        print(d, " ".join(f"{r['epoch']}: {float(r['metrics/mAP50-95(B)'])*100:.1f}/{float(r['metrics/mAP50-95(M)'])*100:.1f}" for r in rows[-3:]))
PY
echo "--- weights backup: copies of best.pt and last.pt of every run go to weights_trained/<run>/ (one older version kept as .prev)"
for d in runs/*/weights; do
  n=$(basename "$(dirname "$d")"); mkdir -p "weights_trained/$n"
  rsync -a --backup --suffix=.prev "$d/" "weights_trained/$n/" && cp "runs/$n/args.yaml" "runs/$n/results.csv" "weights_trained/$n/" 2>/dev/null
done
echo "weights_trained/ now holds: $(find weights_trained -name '*.pt' | wc -l | tr -d ' ') weight files, $(du -sh weights_trained | cut -f1)"
