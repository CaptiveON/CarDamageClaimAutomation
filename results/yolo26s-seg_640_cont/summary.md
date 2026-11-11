| Model | mask AP | mask AP50 | dent | scratch | crack | glass shatter | lamp broken | tire flat |
|---|---|---|---|---|---|---|---|---|
| Mask R-CNN R-50 (paper) | 48.4 | 65.5 | 27.0 | 22.2 | 9.3 | 87.8 | 58.5 | 85.8 |
| DCN+ R-101 (paper) | 57.0 | 77.7 | 40.5 | 34.3 | 16.6 | 89.6 | 70.8 | 90.0 |
| **yolo26s-seg_640_cont (ours, 640px)** | **56.1** | 76.6 | 36.1 | 28.4 | 14.9 | 92.1 | 75.4 | 89.6 |

Box AP 60.7 (DCN+ 60.6, Mask R-CNN 49.2). Speed: 15.8 ms/image on mps (batches of 8), 105.3 ms for a single image on CPU.
