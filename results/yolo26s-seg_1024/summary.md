| Model | mask AP | mask AP50 | dent | scratch | crack | glass shatter | lamp broken | tire flat |
|---|---|---|---|---|---|---|---|---|
| Mask R-CNN R-50 (paper) | 48.4 | 65.5 | 27.0 | 22.2 | 9.3 | 87.8 | 58.5 | 85.8 |
| DCN+ R-101 (paper) | 57.0 | 77.7 | 40.5 | 34.3 | 16.6 | 89.6 | 70.8 | 90.0 |
| **yolo26s-seg_1024 (ours, 1024px)** | **58.4** | 79.4 | 37.2 | 33.0 | 21.9 | 92.4 | 75.8 | 90.3 |

Box AP 62.4 (DCN+ 60.6, Mask R-CNN 49.2). Speed: 21.6 ms/image on mps (batches of 8), 413.9 ms for a single image on CPU.
