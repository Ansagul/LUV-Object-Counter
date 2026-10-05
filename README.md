# Shape Counting with Grayscale vs. Fused CIE-LUV Edge Detection

Classical computer-vision pipeline (OpenCV, no machine learning) that finds and counts objects in an image and sorts them into **triangles, rectangles and polygons (more than four edges)** from their contours.

The project compares two ways of finding edges:

- **Baseline:** grayscale, then Gaussian blur, Canny edges, morphological closing, contours and `approxPolyDP`. The parameters follow a published method.
- **Extension (Fused LUV):** run the same pipeline separately on the L, u and v channels of the CIE-LUV colour space, then merge duplicate detections across channels (IoU-based). The idea is that colour information recovers objects that have low contrast in grayscale.

## Results (6 test images, 70 objects in total)

| Image | Ground truth | Baseline | Fused LUV |
|---|---|---|---|
| Close_Objects | 18 | 4 | **18** |
| Close_Up | 1 | 14 | 17 |
| Far_Objects | 12 | 0 | **12** |
| Low Contrast | 14 | 1 | 21 |
| Optimal High Contrast | 17 | **17** | **17** |
| Shadow_Effect | 8 | 9 | 10 |
| **Total count accuracy** | | 24.3 % | **64.3 %** |

Fused LUV helps on small, close or low-contrast objects. It does not help everywhere. It over-counts on `Close_Up` (a single object with printed text and texture) and does slightly worse than grayscale on `Shadow_Effect`. Both are listed honestly in `results/metrics/`.

"Accuracy" here is count-based (`1 - |detected - truth| / truth`, floored at 0). It does not check that the detected locations are correct.

## Repository layout

```
luv_object_counter.py      full comparison: baseline, L, u, v and fused LUV, with metrics and charts
baseline_object_counter.py minimal grayscale-only version (written for a Raspberry Pi)
images/                    the 6 test images
results/metrics/           CSV tables, confusion matrices, text report, accuracy charts
results/graphs/            comparison heatmaps and charts
results/example_pipeline/  every intermediate step for one image
docs/luv-extension-report.pdf   write-up
```

## Run

```bash
pip install -r requirements.txt

python luv_object_counter.py
# Enter the full image path or folder path: images
# Enter the ground truth CSV path (or press Enter to skip):
```

Outputs are written next to the input: `luv_output/<image>/` (intermediate images) and `luv_metrics/` (tables and charts). Without a ground-truth CSV, the script uses the counts for the six bundled images, which are hardcoded in `KNOWN_COUNT_GROUND_TRUTH`. For your own images, pass a CSV.

On a Raspberry Pi, the lightweight baseline is enough:

```bash
python baseline_object_counter.py   # prompts for a folder of pictures, writes baseline_output/
```

## Limitations

- Ground truth for the bundled images was counted by hand and is hardcoded.
- The thresholds (Canny 10/250, 7x7 closing kernel, 2 % polygon tolerance) are fixed and were not tuned per image.
- Six images is a small evaluation set. Treat the percentages as illustrative.
