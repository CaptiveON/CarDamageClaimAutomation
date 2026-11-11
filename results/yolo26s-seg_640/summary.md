| Model | mask AP | mask AP50 | dent | scratch | crack | glass shatter | lamp broken | tire flat |
|---|---|---|---|---|---|---|---|---|
| Mask R-CNN R-50 (paper) | 48.4 | 65.5 | 27.0 | 22.2 | 9.3 | 87.8 | 58.5 | 85.8 |
| DCN+ R-101 (paper) | 57.0 | 77.7 | 40.5 | 34.3 | 16.6 | 89.6 | 70.8 | 90.0 |
| **yolo26s-seg_640 (ours, 640px)** | **56.7** | 76.1 | 37.0 | 29.5 | 16.3 | 92.7 | 75.2 | 89.8 |

Box AP 61.0 (DCN+ 60.6, Mask R-CNN 49.2). Speed: 15.2 ms/image on mps (batches of 8), 131.5 ms for a single image on CPU.
