| Model | mask AP | mask AP50 | dent | scratch | crack | glass shatter | lamp broken | tire flat |
|---|---|---|---|---|---|---|---|---|
| Mask R-CNN R-50 (paper) | 48.4 | 65.5 | 27.0 | 22.2 | 9.3 | 87.8 | 58.5 | 85.8 |
| DCN+ R-101 (paper) | 57.0 | 77.7 | 40.5 | 34.3 | 16.6 | 89.6 | 70.8 | 90.0 |
| **yolo11s-seg_640 (ours, 640px)** | **55.3** | 74.0 | 33.1 | 28.8 | 16.0 | 92.6 | 72.0 | 89.3 |

Box AP 59.9 (DCN+ 60.6, Mask R-CNN 49.2). Speed: 15.2 ms/image on mps (batches of 8), 106.7 ms for a single image on CPU.
