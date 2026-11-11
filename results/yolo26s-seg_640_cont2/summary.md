| Model | mask AP | mask AP50 | dent | scratch | crack | glass shatter | lamp broken | tire flat |
|---|---|---|---|---|---|---|---|---|
| Mask R-CNN R-50 (paper) | 48.4 | 65.5 | 27.0 | 22.2 | 9.3 | 87.8 | 58.5 | 85.8 |
| DCN+ R-101 (paper) | 57.0 | 77.7 | 40.5 | 34.3 | 16.6 | 89.6 | 70.8 | 90.0 |
| **yolo26s-seg_640_cont2 (ours, 640px)** | **55.9** | 75.5 | 36.1 | 27.8 | 15.8 | 91.4 | 73.9 | 90.2 |

Box AP 60.0 (DCN+ 60.6, Mask R-CNN 49.2). Speed: 15.1 ms/image on mps (batches of 8), 110.9 ms for a single image on CPU.
