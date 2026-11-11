# Model cards: every trained weight in this repository

All damage models were scored once on the official CarDD test split (374 photos, 785 damages) with pycocotools. "Small, medium, large" are mask AP by object size (COCO's size bands). Speed is one photo at the model's input size, batch 1: a laptop CPU (Apple M2 Max, no GPU used) and the laptop GPU (Metal). Each folder also holds `args.yaml` (every training setting, as Ultralytics recorded it) and `results.csv` (losses and validation scores per epoch). Licence: AGPL-3.0, trained on a research-only dataset; see the main README.

Class order for all damage models: `dent`, `scratch`, `crack`, `glass shatter`, `lamp broken`, `tire flat`.

## `yolo26s-seg_1024/best.pt`

The best run: full-resolution training from the start.

| | |
|---|---|
| Architecture | YOLO26s-seg (Ultralytics), instance segmentation, 11.5 M parameters, 22 MB |
| Started from | COCO-pretrained yolo26s-seg.pt |
| Input size | 1024 px |
| Epochs, batch | 60, 4 |
| Optimiser | auto (AdamW, lr 0.001) |
| Training time | 12.7 h on an Apple M2 Max (Metal); best validation mask AP 60.0 |
| **Test mask AP** | **58.4** (AP50 79.4) |
| Test box AP | 62.4 (AP50 79.5) |
| Mask AP per class | dent 37.2, scratch 33.0, crack 21.9, glass 92.4, lamp 75.8, tyre 90.3 |
| Box AP per class | dent 38.6, scratch 40.0, crack 37.5, glass 92.7, lamp 76.0, tyre 89.6 |
| Mask AP by size | small 24.5, medium 26.5, large 59.8 |
| Speed | 414 ms per photo on CPU, 21.6 ms on the laptop GPU |

## `yolo26s-seg_640/best.pt`

The 640 px baseline of the same model.

| | |
|---|---|
| Architecture | YOLO26s-seg (Ultralytics), instance segmentation, 11.5 M parameters, 22 MB |
| Started from | COCO-pretrained yolo26s-seg.pt |
| Input size | 640 px |
| Epochs, batch | 40, 8 |
| Optimiser | auto (AdamW, lr 0.001) |
| Training time | 4.3 h on an Apple M2 Max (Metal); best validation mask AP 58.6 |
| **Test mask AP** | **56.7** (AP50 76.1) |
| Test box AP | 61.0 (AP50 77.9) |
| Mask AP per class | dent 37.0, scratch 29.5, crack 16.3, glass 92.7, lamp 75.2, tyre 89.8 |
| Box AP per class | dent 40.4, scratch 36.4, crack 30.6, glass 91.4, lamp 76.5, tyre 90.5 |
| Mask AP by size | small 13.1, medium 28.5, large 58.6 |
| Speed | 132 ms per photo on CPU, 15.2 ms on the laptop GPU |

## `yolo26m-seg_640/best.pt`

The medium model, half the epochs.

| | |
|---|---|
| Architecture | YOLO26m-seg (Ultralytics), instance segmentation, 27.1 M parameters, 52 MB |
| Started from | COCO-pretrained yolo26m-seg.pt |
| Input size | 640 px |
| Epochs, batch | 20, 8 |
| Optimiser | auto (AdamW, lr 0.001) |
| Training time | 3.1 h on an Apple M2 Max (Metal); best validation mask AP 59.3 |
| **Test mask AP** | **56.1** (AP50 76.3) |
| Test box AP | 60.6 (AP50 77.3) |
| Mask AP per class | dent 37.4, scratch 31.4, crack 15.3, glass 92.0, lamp 72.9, tyre 87.6 |
| Box AP per class | dent 41.3, scratch 40.4, crack 30.3, glass 91.2, lamp 72.8, tyre 87.7 |
| Mask AP by size | small 17.9, medium 24.8, large 56.8 |
| Speed | 295 ms per photo on CPU, 23.4 ms on the laptop GPU |

## `yolo26s-seg_640_cont/best.pt`

A finished run continued for 10 gentle epochs.

| | |
|---|---|
| Architecture | YOLO26s-seg (Ultralytics), instance segmentation, 11.5 M parameters, 22 MB |
| Started from | continued from yolo26s-seg_640 |
| Input size | 640 px |
| Epochs, batch | 10, 8 |
| Optimiser | AdamW, lr 0.0005 |
| Training time | 0.9 h on an Apple M2 Max (Metal); best validation mask AP 57.9 |
| **Test mask AP** | **56.1** (AP50 76.6) |
| Test box AP | 60.7 (AP50 77.5) |
| Mask AP per class | dent 36.1, scratch 28.4, crack 14.9, glass 92.1, lamp 75.4, tyre 89.6 |
| Box AP per class | dent 39.3, scratch 36.1, crack 28.5, glass 93.5, lamp 75.8, tyre 91.1 |
| Mask AP by size | small 11.1, medium 28.7, large 57.9 |
| Speed | 105 ms per photo on CPU, 15.8 ms on the laptop GPU |

## `yolo26s-seg_640_cont2/best.pt`

That continuation continued for 10 more.

| | |
|---|---|
| Architecture | YOLO26s-seg (Ultralytics), instance segmentation, 11.5 M parameters, 22 MB |
| Started from | continued from yolo26s-seg_640_cont |
| Input size | 640 px |
| Epochs, batch | 10, 8 |
| Optimiser | AdamW, lr 0.0002 |
| Training time | 0.9 h on an Apple M2 Max (Metal); best validation mask AP 57.5 |
| **Test mask AP** | **55.9** (AP50 75.5) |
| Test box AP | 60.0 (AP50 76.3) |
| Mask AP per class | dent 36.1, scratch 27.8, crack 15.8, glass 91.4, lamp 73.9, tyre 90.2 |
| Box AP per class | dent 39.1, scratch 35.5, crack 27.6, glass 93.8, lamp 74.1, tyre 89.7 |
| Mask AP by size | small 10.3, medium 29.2, large 57.9 |
| Speed | 111 ms per photo on CPU, 15.1 ms on the laptop GPU |

## `yolo11s-seg_640/best.pt`

The previous YOLO generation, same recipe.

| | |
|---|---|
| Architecture | YOLO11s-seg (Ultralytics), instance segmentation, 10.1 M parameters, 20 MB |
| Started from | COCO-pretrained yolo11s-seg.pt |
| Input size | 640 px |
| Epochs, batch | 40, 8 |
| Optimiser | auto (AdamW, lr 0.001) |
| Training time | 2.3 h on an Apple M2 Max (Metal); best validation mask AP 56.7 |
| **Test mask AP** | **55.3** (AP50 74.0) |
| Test box AP | 59.9 (AP50 75.6) |
| Mask AP per class | dent 33.1, scratch 28.8, crack 16.0, glass 92.6, lamp 72.0, tyre 89.3 |
| Box AP per class | dent 37.3, scratch 36.0, crack 27.4, glass 94.7, lamp 73.9, tyre 90.2 |
| Mask AP by size | small 24.2, medium 25.5, large 56.5 |
| Speed | 107 ms per photo on CPU, 15.2 ms on the laptop GPU |

## `yolo11s-seg_640_bal/best.pt`

Crack boost: crack photos shown 3x per epoch plus copy-paste.

| | |
|---|---|
| Architecture | YOLO11s-seg (Ultralytics), instance segmentation, 10.1 M parameters, 20 MB |
| Started from | continued from yolo11s-seg_640 |
| Input size | 640 px |
| Epochs, batch | 20, 8 |
| Optimiser | SGD, lr 0.003 · copy_paste 0.3 (mixup), crack photos x3 |
| Training time | 1.4 h on an Apple M2 Max (Metal); best validation mask AP 57.1 |
| **Test mask AP** | **54.5** (AP50 73.9) |
| Test box AP | 59.6 (AP50 74.8) |
| Mask AP per class | dent 31.3, scratch 28.5, crack 17.8, glass 91.8, lamp 73.4, tyre 84.5 |
| Box AP per class | dent 35.1, scratch 35.1, crack 31.7, glass 94.0, lamp 75.9, tyre 85.9 |
| Mask AP by size | small 18.9, medium 27.0, large 55.7 |
| Speed | 131 ms per photo on CPU, 15.5 ms on the laptop GPU |

## `yolo11s-seg_1024/best.pt`

A finished 640 px model continued at 1024 px.

| | |
|---|---|
| Architecture | YOLO11s-seg (Ultralytics), instance segmentation, 10.1 M parameters, 20 MB |
| Started from | continued from yolo11s-seg_640 |
| Input size | 1024 px |
| Epochs, batch | 30, 4 |
| Optimiser | auto (AdamW, lr 0.001) |
| Training time | 3.7 h on an Apple M2 Max (Metal); best validation mask AP 56.3 |
| **Test mask AP** | **53.3** (AP50 71.0) |
| Test box AP | 57.1 (AP50 74.0) |
| Mask AP per class | dent 31.0, scratch 28.3, crack 16.3, glass 86.9, lamp 70.4, tyre 86.9 |
| Box AP per class | dent 35.1, scratch 33.8, crack 29.3, glass 86.2, lamp 70.2, tyre 88.4 |
| Mask AP by size | small 18.6, medium 24.3, large 54.0 |
| Speed | 382 ms per photo on CPU, 21.3 ms on the laptop GPU |

## `parts_yolo26s_640/best.pt` (car parts, for panel names)

| | |
|---|---|
| Architecture | YOLO26s-seg (Ultralytics), instance segmentation, 23 car-part classes, 22 MB |
| Dataset | Ultralytics Carparts segmentation (Roboflow "car-seg" by Gianmarco Russo, CC BY 4.0): 3,156 train, 401 val, 276 test photos |
| Started from | COCO-pretrained yolo26s-seg.pt |
| Input size, epochs, batch | 640 px, 40, 8; horizontal flip disabled because the labels distinguish left from right |
| Training time | 5.5 h on an Apple M2 Max (Metal) |
| **Test mask mAP50-95** | **70.4** (mAP50 86.5) |
| Test box mAP50-95 | 72.6 (mAP50 86.2) |
| On CarDD close-ups | names the panel for about one damage in seven at confidence 0.6 (measured by `src/parts_check.py`); good on whole-car photos |
