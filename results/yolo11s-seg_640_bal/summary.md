| Model | mask AP | mask AP50 | dent | scratch | crack | glass shatter | lamp broken | tire flat |
|---|---|---|---|---|---|---|---|---|
| Mask R-CNN R-50 (paper) | 48.4 | 65.5 | 27.0 | 22.2 | 9.3 | 87.8 | 58.5 | 85.8 |
| DCN+ R-101 (paper) | 57.0 | 77.7 | 40.5 | 34.3 | 16.6 | 89.6 | 70.8 | 90.0 |
| **yolo11s-seg_640_bal (ours, 640px)** | **54.5** | 73.9 | 31.3 | 28.5 | 17.8 | 91.8 | 73.4 | 84.5 |

Box AP 59.6 (DCN+ 60.6, Mask R-CNN 49.2). Speed: 15.5 ms/image on mps (batches of 8), 130.8 ms for a single image on CPU.
