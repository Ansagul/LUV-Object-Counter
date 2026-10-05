# -------------------- IMPORTING LIBRARIES --------------------
from pathlib import Path
import csv
import math
import sys
import cv2
import numpy as np


# -------------------- TAKING IMAGE OR FOLDER NAME FROM USER --------------------
input_name = input("Enter the full image path or folder path: ")
input_path = Path(input_name)
ground_truth_name = input("Enter the ground truth CSV path (or press Enter to skip): ").strip()
ground_truth_path = Path(ground_truth_name) if ground_truth_name else None


# -------------------- READING IMAGE FILES --------------------
if input_path.is_dir():
    image_files = []
    image_files += sorted(input_path.glob("*.jpg"))
    image_files += sorted(input_path.glob("*.jpeg"))
    image_files += sorted(input_path.glob("*.png"))
    image_files += sorted(input_path.glob("*.bmp"))

    if len(image_files) == 0:
        sys.exit(f"Error: No images were found in folder '{input_path}'.")
else:
    image_files = [input_path]


if ground_truth_path is not None and ground_truth_path.exists() is False:
    sys.exit(f"Error: Ground truth file '{ground_truth_path}' was not found.")


metrics_output_folder = (input_path if input_path.is_dir() else input_path.parent) / "luv_metrics"
metrics_output_folder.mkdir(parents=True, exist_ok=True)


KNOWN_COUNT_GROUND_TRUTH = {
    "closeobjects": {"total": 18, "triangles": 4, "rectangles": 4, "polygons": 10},
    "closeup": {"total": 1, "triangles": 0, "rectangles": 1, "polygons": 0},
    "farobjects": {"total": 12, "triangles": 3, "rectangles": 5, "polygons": 4},
    "lowcontrast": {"total": 14, "triangles": 3, "rectangles": 4, "polygons": 7},
    "optimalhighcontrast": {"total": 17, "triangles": 4, "rectangles": 4, "polygons": 9},
    "shadoweffect": {"total": 8, "triangles": 1, "rectangles": 3, "polygons": 4},
}
SHAPE_LABELS = ["triangle", "rectangle", "polygon"]
UNKNOWN_SHAPE_LABEL = "unknown"
UNMATCHED_SHAPE_LABEL = "unmatched"
METHOD_CHART_LABELS = {
    "Grayscale": "Gray",
    "L channel": "L",
    "u channel": "u",
    "v channel": "v",
    "Fused LUV": "LUV",
}
METHOD_CHART_COLORS = {
    "Grayscale": (120, 120, 120),
    "L channel": (240, 130, 60),
    "u channel": (80, 175, 76),
    "v channel": (76, 110, 220),
    "Fused LUV": (40, 170, 210),
}
IMAGE_CHART_LABELS = {
    "closeobjects": "CloseObj",
    "closeup": "CloseUp",
    "farobjects": "Far",
    "lowcontrast": "Low",
    "optimalhighcontrast": "High",
    "shadoweffect": "Shadow",
}


# -------------------- DEFINING FUNCTIONS --------------------
# Function: normalize_lookup_name
# Purpose: convert image names into a consistent lookup key.
def normalize_lookup_name(name):
    return "".join(character.lower() for character in name if character.isalnum())


# Function: normalize_shape_label
# Purpose: map shape labels into a stable set for tables and confusion matrices.
def normalize_shape_label(shape_name, include_unknown):
    if shape_name in SHAPE_LABELS:
        return shape_name
    if include_unknown:
        return UNKNOWN_SHAPE_LABEL
    return None


# Function: process_channel
# Purpose: full detection pipeline on one image channel.
# Logic:
# 1. blur the channel
# 2. detect edges
# 3. close small gaps
# 4. find contours
# 5. approximate each contour to a simple shape
# 6. store the detected object information
def process_channel(
    channel_image,
    original_image,
    draw_color,
    channel_name,
    low_threshold=10,
    high_threshold=250,
    kernel_size=(7, 7),
):
    # This reduces noise in the image
    # Parameter used here: GaussianBlur(channel, (3, 3), 0)
    blurred = cv2.GaussianBlur(channel_image, (3, 3), 0)

    # This finds the boundaries of the objects
    # The thresholds can be changed to make detection more sensitive
    edged = cv2.Canny(blurred, low_threshold, high_threshold)

    # This joins broken edges so the shapes can be detected more clearly
    # The kernel size can also be changed for different sensitivity levels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # Contours are the outlines of the objects
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    result = original_image.copy()

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)

        # Parameter used here: 0.02 * perimeter (reduces the number of points in the contour approximation)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) < 3:
            continue

        if len(approx) == 3:
            shape_name = "triangle"
        elif len(approx) == 4:
            shape_name = "rectangle"
        else:
            shape_name = "polygon"

        x, y, w, h = cv2.boundingRect(approx)
        area = cv2.contourArea(contour)

        detections.append(
            {
                "shape": shape_name,
                "box": (x, y, w, h),
                "area": area,
                "contour": approx,
                "channel": channel_name,
            }
        )

        cv2.drawContours(result, [approx], -1, draw_color, 2)
        cv2.putText(result, shape_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 1)

    return detections, blurred, edged, closed, result


# Function: enhance_luv_channel
# Purpose: increase the contrast of one L, u, or v channel before detection.
# Logic:
# 1. stretch the channel values to the full 0-255 range
# 2. apply CLAHE to make weak edges more visible
def enhance_luv_channel(channel_image):
    normalized = cv2.normalize(channel_image, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)
    return enhanced


# Function: box_iou
# Purpose: measure how much two bounding boxes overlap.
# Logic:
# 1. find the common overlapping area
# 2. compute the total union area
# 3. return overlap / union
# If the value is high, both detections are probably the same object.
def box_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)

    if right <= left or bottom <= top:
        return 0

    intersection = (right - left) * (bottom - top)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    return intersection / union


# Function: same_object
# Purpose: decide whether two detections refer to the same real object.
# Logic:
# 1. check if their boxes overlap enough
# 2. if not, check whether their centers are still very close
# 3. if either condition is true, treat them as the same object
def same_object(detection1, detection2):
    box1 = detection1["box"]
    box2 = detection2["box"]

    overlap = box_iou(box1, box2)
    if overlap > 0.20:
        return True

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    center1_x = x1 + (w1 / 2)
    center1_y = y1 + (h1 / 2)
    center2_x = x2 + (w2 / 2)
    center2_y = y2 + (h2 / 2)

    distance = math.hypot(center1_x - center2_x, center1_y - center2_y)
    limit = max(w1, h1, w2, h2) * 0.35

    return distance < limit


# Function: merge_unique_objects
# Purpose: combine detections from the L, u, and v channels.
# Logic:
# 1. compare every new detection with already saved detections
# 2. if it matches an existing object, keep only one copy
# 3. if the new one has a larger area, replace the old one
# 4. if no match is found, add it as a new unique object
def merge_unique_objects(all_detections):
    # There is no exact merge formula here, so overlapping or very close detections
    # are treated as the same object.
    unique_objects = []

    for detection in all_detections:
        already_present = False

        for i in range(len(unique_objects)):
            if same_object(detection, unique_objects[i]):
                already_present = True

                # Keep the larger detection because it usually gives a better outline
                if detection["area"] > unique_objects[i]["area"]:
                    unique_objects[i] = detection
                break

        if already_present is False:
            unique_objects.append(detection)

    return unique_objects


# Function: count_shapes
# Purpose: count how many triangles, rectangles, and polygons were found.
# Logic:
# 1. read the saved shape label of every detection
# 2. increase the correct counter
def count_shapes(detections):
    triangles = 0
    rectangles = 0
    polygons = 0

    for detection in detections:
        if detection["shape"] == "triangle":
            triangles += 1
        elif detection["shape"] == "rectangle":
            rectangles += 1
        else:
            polygons += 1

    return triangles, rectangles, polygons


# Function: create_summary_row
# Purpose: store one comparison row for the final table.
# Logic:
# 1. take the image name and method name
# 2. store total detected objects
# 3. store how many triangles, rectangles, and polygons were found
def create_summary_row(image_name, method_name, total, triangles, rectangles, polygons):
    return {
        "Image": image_name,
        "Method": method_name,
        "Total Objects": total,
        "Triangles": triangles,
        "Rectangles": rectangles,
        "Polygons": polygons,
    }


# Function: create_channel_summary_row
# Purpose: store one row for a single L, u, or v channel.
def create_channel_summary_row(image_name, channel_name, total, triangles, rectangles, polygons):
    return {
        "Image": image_name,
        "Channel": channel_name,
        "Total Objects": total,
        "Triangles": triangles,
        "Rectangles": rectangles,
        "Polygons": polygons,
    }


# Function: create_total_comparison_row
# Purpose: store one row for the short total-comparison table.
# Logic:
# 1. save the image index
# 2. save the grayscale total
# 3. save the merged LUV total
def create_total_comparison_row(image_index, grayscale_total, luv_total):
    return {
        "Image Index": image_index,
        "Grayscale Total": grayscale_total,
        "Merged LUV Total": luv_total,
    }


# Function: load_ground_truth
# Purpose: read optional ground-truth annotations from a CSV file.
# Supported box formats:
# 1. image,x,y,w,h[,shape]
# 2. image,x1,y1,x2,y2[,shape]
def load_ground_truth(csv_path):
    if csv_path is None:
        return {}

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            sys.exit("Error: The ground truth CSV file is empty.")

        headers = {field.strip().lower(): field for field in reader.fieldnames}
        if "image" not in headers and "image_name" not in headers and "filename" not in headers and "file" not in headers:
            sys.exit("Error: Ground truth CSV must contain an image column such as 'image'.")

        has_xywh = {"x", "y", "w", "h"}.issubset(headers.keys())
        has_corners = {"x1", "y1", "x2", "y2"}.issubset(headers.keys())
        if has_xywh is False and has_corners is False:
            sys.exit("Error: Ground truth CSV must contain x,y,w,h or x1,y1,x2,y2 columns.")

        ground_truth_by_image = {}

        for row in reader:
            image_name = read_csv_value(row, headers, "image", "image_name", "filename", "file")
            if image_name == "":
                continue

            if has_xywh:
                x = int(round(float(read_csv_value(row, headers, "x"))))
                y = int(round(float(read_csv_value(row, headers, "y"))))
                w = int(round(float(read_csv_value(row, headers, "w"))))
                h = int(round(float(read_csv_value(row, headers, "h"))))
            else:
                x1 = int(round(float(read_csv_value(row, headers, "x1"))))
                y1 = int(round(float(read_csv_value(row, headers, "y1"))))
                x2 = int(round(float(read_csv_value(row, headers, "x2"))))
                y2 = int(round(float(read_csv_value(row, headers, "y2"))))
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1

            shape_name = read_csv_value(row, headers, "shape", "label", "class").lower()
            if shape_name == "":
                shape_name = None

            ground_truth_by_image.setdefault(image_name, []).append(
                {
                    "shape": shape_name,
                    "box": (x, y, w, h),
                    "area": w * h,
                    "channel": "ground_truth",
                }
            )

    return ground_truth_by_image


# Function: read_csv_value
# Purpose: fetch one CSV value using a list of possible column names.
def read_csv_value(row, headers, *candidate_names):
    for candidate_name in candidate_names:
        if candidate_name in headers:
            raw_value = row[headers[candidate_name]]
            if raw_value is None:
                return ""
            return raw_value.strip()
    return ""


# Function: get_ground_truth_for_image
# Purpose: find annotations for one image using either the full name or stem.
def get_ground_truth_for_image(image_path, ground_truth_by_image):
    if image_path.name in ground_truth_by_image:
        return ground_truth_by_image[image_path.name]
    if image_path.stem in ground_truth_by_image:
        return ground_truth_by_image[image_path.stem]
    return []


# Function: get_known_count_ground_truth_for_image
# Purpose: find the correct object counts for one known test image.
def get_known_count_ground_truth_for_image(image_path):
    image_name_key = normalize_lookup_name(image_path.name)
    image_stem_key = normalize_lookup_name(image_path.stem)

    if image_name_key in KNOWN_COUNT_GROUND_TRUTH:
        return KNOWN_COUNT_GROUND_TRUTH[image_name_key]
    if image_stem_key in KNOWN_COUNT_GROUND_TRUTH:
        return KNOWN_COUNT_GROUND_TRUTH[image_stem_key]
    return None


# Function: safe_divide
# Purpose: avoid division-by-zero errors in metric calculations.
def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


# Function: match_detections
# Purpose: greedily pair detections using the highest IoU matches first.
def match_detections(reference_detections, candidate_detections, iou_threshold):
    possible_matches = []

    for reference_index, reference_detection in enumerate(reference_detections):
        for candidate_index, candidate_detection in enumerate(candidate_detections):
            overlap = box_iou(reference_detection["box"], candidate_detection["box"])
            if overlap >= iou_threshold:
                possible_matches.append((overlap, reference_index, candidate_index))

    possible_matches.sort(reverse=True)

    used_reference_indexes = set()
    used_candidate_indexes = set()
    matches = []

    for overlap, reference_index, candidate_index in possible_matches:
        if reference_index in used_reference_indexes or candidate_index in used_candidate_indexes:
            continue

        used_reference_indexes.add(reference_index)
        used_candidate_indexes.add(candidate_index)

        matches.append(
            {
                "reference_index": reference_index,
                "candidate_index": candidate_index,
                "iou": overlap,
                "same_shape": reference_detections[reference_index].get("shape")
                == candidate_detections[candidate_index].get("shape"),
            }
        )

    return matches


# Function: compare_detection_sets
# Purpose: compare baseline and fused-LUV detections without ground truth.
def compare_detection_sets(baseline_detections, fused_detections):
    matches = match_detections(baseline_detections, fused_detections, 0.20)
    matched_count = len(matches)
    baseline_total = len(baseline_detections)
    fused_total = len(fused_detections)
    union_count = baseline_total + fused_total - matched_count

    if union_count == 0:
        agreement_rate = 1.0
    else:
        agreement_rate = matched_count / union_count

    mean_iou = safe_divide(sum(match["iou"] for match in matches), matched_count)
    shape_agreement = None
    if matched_count > 0:
        shape_agreement = sum(1 for match in matches if match["same_shape"]) / matched_count

    return {
        "baseline_total": baseline_total,
        "fused_total": fused_total,
        "matched_count": matched_count,
        "baseline_only": baseline_total - matched_count,
        "luv_only": fused_total - matched_count,
        "agreement_rate": agreement_rate,
        "mean_iou": mean_iou,
        "shape_agreement": shape_agreement,
        "count_delta": fused_total - baseline_total,
    }


# Function: evaluate_against_ground_truth
# Purpose: compute standard detection metrics for one method on one image.
def evaluate_against_ground_truth(detections, ground_truth_detections):
    matches = match_detections(ground_truth_detections, detections, 0.50)
    true_positives = len(matches)
    predicted_total = len(detections)
    ground_truth_total = len(ground_truth_detections)
    false_positives = predicted_total - true_positives
    false_negatives = ground_truth_total - true_positives
    precision = safe_divide(true_positives, true_positives + false_positives)
    recall = safe_divide(true_positives, true_positives + false_negatives)
    f1_score = safe_divide(2 * precision * recall, precision + recall)
    iou_sum = sum(match["iou"] for match in matches)
    mean_iou = safe_divide(iou_sum, true_positives)
    shape_match_count = 0
    shape_correct = 0
    for match in matches:
        if ground_truth_detections[match["reference_index"]].get("shape") is not None:
            shape_match_count += 1
            if match["same_shape"]:
                shape_correct += 1

    shape_accuracy = None
    if shape_match_count > 0:
        shape_accuracy = shape_correct / shape_match_count

    return {
        "gt_total": ground_truth_total,
        "pred_total": predicted_total,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "mean_iou": mean_iou,
        "iou_sum": iou_sum,
        "shape_accuracy": shape_accuracy,
        "shape_correct": shape_correct,
        "shape_match_count": shape_match_count,
        "count_error": predicted_total - ground_truth_total,
        "absolute_count_error": abs(predicted_total - ground_truth_total),
    }


# Function: evaluate_counts_against_ground_truth
# Purpose: compare predicted counts against the known correct counts.
def evaluate_counts_against_ground_truth(total, triangles, rectangles, polygons, ground_truth_counts):
    total_error = total - ground_truth_counts["total"]
    triangle_error = triangles - ground_truth_counts["triangles"]
    rectangle_error = rectangles - ground_truth_counts["rectangles"]
    polygon_error = polygons - ground_truth_counts["polygons"]
    shape_abs_error = abs(triangle_error) + abs(rectangle_error) + abs(polygon_error)

    return {
        "gt_total": ground_truth_counts["total"],
        "pred_total": total,
        "total_error": total_error,
        "absolute_total_error": abs(total_error),
        "gt_triangles": ground_truth_counts["triangles"],
        "pred_triangles": triangles,
        "triangle_error": triangle_error,
        "gt_rectangles": ground_truth_counts["rectangles"],
        "pred_rectangles": rectangles,
        "rectangle_error": rectangle_error,
        "gt_polygons": ground_truth_counts["polygons"],
        "pred_polygons": polygons,
        "polygon_error": polygon_error,
        "shape_abs_error": shape_abs_error,
        "exact_total_match": total_error == 0,
        "exact_shape_match": triangle_error == 0 and rectangle_error == 0 and polygon_error == 0,
    }


# Function: create_empty_confusion_matrix
# Purpose: create a square confusion matrix with stable labels.
def create_empty_confusion_matrix(include_unknown=False, include_unmatched=True):
    labels = list(SHAPE_LABELS)
    if include_unknown:
        labels.append(UNKNOWN_SHAPE_LABEL)
    if include_unmatched:
        labels.append(UNMATCHED_SHAPE_LABEL)

    return {row_label: {column_label: 0 for column_label in labels} for row_label in labels}


# Function: build_confusion_matrix
# Purpose: build a shape confusion matrix from matched detections.
def build_confusion_matrix(
    reference_detections,
    candidate_detections,
    iou_threshold,
    include_unknown=False,
    include_unmatched=True,
):
    confusion_matrix = create_empty_confusion_matrix(include_unknown, include_unmatched)
    matches = match_detections(reference_detections, candidate_detections, iou_threshold)
    matched_reference_indexes = set()
    matched_candidate_indexes = set()

    for match in matches:
        reference_index = match["reference_index"]
        candidate_index = match["candidate_index"]
        matched_reference_indexes.add(reference_index)
        matched_candidate_indexes.add(candidate_index)

        reference_label = normalize_shape_label(reference_detections[reference_index].get("shape"), include_unknown)
        candidate_label = normalize_shape_label(candidate_detections[candidate_index].get("shape"), include_unknown)
        if reference_label is None or candidate_label is None:
            continue

        confusion_matrix[reference_label][candidate_label] += 1

    if include_unmatched:
        for reference_index, reference_detection in enumerate(reference_detections):
            if reference_index in matched_reference_indexes:
                continue

            reference_label = normalize_shape_label(reference_detection.get("shape"), include_unknown)
            if reference_label is None:
                continue
            confusion_matrix[reference_label][UNMATCHED_SHAPE_LABEL] += 1

        for candidate_index, candidate_detection in enumerate(candidate_detections):
            if candidate_index in matched_candidate_indexes:
                continue

            candidate_label = normalize_shape_label(candidate_detection.get("shape"), include_unknown)
            if candidate_label is None:
                continue
            confusion_matrix[UNMATCHED_SHAPE_LABEL][candidate_label] += 1

    return confusion_matrix


# Function: accumulate_confusion_matrix
# Purpose: add one confusion matrix into another.
def accumulate_confusion_matrix(total_confusion_matrix, partial_confusion_matrix):
    for row_label in total_confusion_matrix:
        for column_label in total_confusion_matrix[row_label]:
            total_confusion_matrix[row_label][column_label] += partial_confusion_matrix[row_label][column_label]


# Function: create_confusion_matrix_rows
# Purpose: flatten one confusion matrix into CSV-friendly rows.
def create_confusion_matrix_rows(matrix_name, row_label_name, confusion_matrix):
    rows = []
    column_labels = list(next(iter(confusion_matrix.values())).keys())

    for row_label, row_values in confusion_matrix.items():
        row = {
            "Matrix": matrix_name,
            row_label_name: row_label,
        }
        for column_label in column_labels:
            row[column_label] = row_values[column_label]
        rows.append(row)

    return rows


# Function: sanitize_filename_part
# Purpose: convert matrix names into safe CSV filenames.
def sanitize_filename_part(text):
    filename_characters = []
    previous_was_separator = False

    for character in text.lower():
        if character.isalnum():
            filename_characters.append(character)
            previous_was_separator = False
        elif previous_was_separator is False:
            filename_characters.append("_")
            previous_was_separator = True

    sanitized_name = "".join(filename_characters).strip("_")
    if sanitized_name == "":
        return "confusion_matrix"
    return sanitized_name


# Function: create_method_comparison_row
# Purpose: store the direct grayscale-vs-LUV comparison metrics.
def create_method_comparison_row(image_name, comparison_metrics):
    return {
        "Image": image_name,
        "Baseline Total": comparison_metrics["baseline_total"],
        "Fused LUV Total": comparison_metrics["fused_total"],
        "Matched Objects": comparison_metrics["matched_count"],
        "Baseline Only": comparison_metrics["baseline_only"],
        "LUV Only": comparison_metrics["luv_only"],
        "Agreement Rate": comparison_metrics["agreement_rate"],
        "Mean Match IoU": comparison_metrics["mean_iou"],
        "Shape Agreement": comparison_metrics["shape_agreement"],
        "Count Delta": comparison_metrics["count_delta"],
    }


# Function: create_channel_match_row
# Purpose: store grayscale-vs-single-channel match metrics.
def create_channel_match_row(image_name, channel_name, comparison_metrics):
    return {
        "Image": image_name,
        "Channel": channel_name,
        "Baseline Total": comparison_metrics["baseline_total"],
        "Channel Total": comparison_metrics["fused_total"],
        "Matched Objects": comparison_metrics["matched_count"],
        "Baseline Only": comparison_metrics["baseline_only"],
        "Channel Only": comparison_metrics["luv_only"],
        "Agreement Rate": comparison_metrics["agreement_rate"],
        "Mean Match IoU": comparison_metrics["mean_iou"],
        "Shape Agreement": comparison_metrics["shape_agreement"],
        "Count Delta": comparison_metrics["count_delta"],
    }


# Function: create_ground_truth_row
# Purpose: store one supervised evaluation row.
def create_ground_truth_row(image_name, method_name, evaluation_metrics):
    return {
        "Image": image_name,
        "Method": method_name,
        "GT Objects": evaluation_metrics["gt_total"],
        "Detected Objects": evaluation_metrics["pred_total"],
        "TP": evaluation_metrics["tp"],
        "FP": evaluation_metrics["fp"],
        "FN": evaluation_metrics["fn"],
        "Precision": evaluation_metrics["precision"],
        "Recall": evaluation_metrics["recall"],
        "F1": evaluation_metrics["f1"],
        "Mean IoU": evaluation_metrics["mean_iou"],
        "Shape Accuracy": evaluation_metrics["shape_accuracy"],
        "Count Error": evaluation_metrics["count_error"],
        "Absolute Count Error": evaluation_metrics["absolute_count_error"],
    }


# Function: create_count_ground_truth_row
# Purpose: store one count-based evaluation row.
def create_count_ground_truth_row(image_name, method_name, evaluation_metrics):
    return {
        "Image": image_name,
        "Method": method_name,
        "GT Total": evaluation_metrics["gt_total"],
        "Detected Total": evaluation_metrics["pred_total"],
        "Total Error": evaluation_metrics["total_error"],
        "Absolute Total Error": evaluation_metrics["absolute_total_error"],
        "GT Triangles": evaluation_metrics["gt_triangles"],
        "Detected Triangles": evaluation_metrics["pred_triangles"],
        "Triangle Error": evaluation_metrics["triangle_error"],
        "GT Rectangles": evaluation_metrics["gt_rectangles"],
        "Detected Rectangles": evaluation_metrics["pred_rectangles"],
        "Rectangle Error": evaluation_metrics["rectangle_error"],
        "GT Polygons": evaluation_metrics["gt_polygons"],
        "Detected Polygons": evaluation_metrics["pred_polygons"],
        "Polygon Error": evaluation_metrics["polygon_error"],
        "Shape Abs Error": evaluation_metrics["shape_abs_error"],
        "Exact Total Match": "Yes" if evaluation_metrics["exact_total_match"] else "No",
        "Exact Shape Match": "Yes" if evaluation_metrics["exact_shape_match"] else "No",
    }


# Function: create_overall_ground_truth_row
# Purpose: summarize one method across every image that has ground truth.
def create_overall_ground_truth_row(method_name, overall_metrics):
    precision = safe_divide(overall_metrics["tp"], overall_metrics["tp"] + overall_metrics["fp"])
    recall = safe_divide(overall_metrics["tp"], overall_metrics["tp"] + overall_metrics["fn"])
    f1_score = safe_divide(2 * precision * recall, precision + recall)
    mean_iou = safe_divide(overall_metrics["iou_sum"], overall_metrics["tp"])
    shape_accuracy = None
    if overall_metrics["shape_match_count"] > 0:
        shape_accuracy = overall_metrics["shape_correct"] / overall_metrics["shape_match_count"]

    return {
        "Image": "ALL IMAGES",
        "Method": method_name,
        "GT Objects": overall_metrics["gt_total"],
        "Detected Objects": overall_metrics["pred_total"],
        "TP": overall_metrics["tp"],
        "FP": overall_metrics["fp"],
        "FN": overall_metrics["fn"],
        "Precision": precision,
        "Recall": recall,
        "F1": f1_score,
        "Mean IoU": mean_iou,
        "Shape Accuracy": shape_accuracy,
        "Count Error": overall_metrics["pred_total"] - overall_metrics["gt_total"],
        "Absolute Count Error": overall_metrics["absolute_count_error_total"],
    }


# Function: create_overall_count_ground_truth_row
# Purpose: summarize one method across every image with known correct counts.
def create_overall_count_ground_truth_row(method_name, overall_metrics):
    return {
        "Image": "ALL IMAGES",
        "Method": method_name,
        "GT Total": overall_metrics["gt_total"],
        "Detected Total": overall_metrics["pred_total"],
        "Total Error": overall_metrics["total_error"],
        "Absolute Total Error": overall_metrics["absolute_total_error"],
        "GT Triangles": overall_metrics["gt_triangles"],
        "Detected Triangles": overall_metrics["pred_triangles"],
        "Triangle Error": overall_metrics["triangle_error"],
        "GT Rectangles": overall_metrics["gt_rectangles"],
        "Detected Rectangles": overall_metrics["pred_rectangles"],
        "Rectangle Error": overall_metrics["rectangle_error"],
        "GT Polygons": overall_metrics["gt_polygons"],
        "Detected Polygons": overall_metrics["pred_polygons"],
        "Polygon Error": overall_metrics["polygon_error"],
        "Shape Abs Error": overall_metrics["shape_abs_error"],
        "Exact Total Match": f"{overall_metrics['exact_total_match_count']}/{overall_metrics['image_count']}",
        "Exact Shape Match": f"{overall_metrics['exact_shape_match_count']}/{overall_metrics['image_count']}",
    }


# Function: create_accuracy_matrix_row
# Purpose: build one poster-ready baseline-vs-LUV accuracy row.
def create_accuracy_matrix_row(
    image_name,
    gt_total,
    baseline_total,
    baseline_abs_error,
    luv_total,
    luv_abs_error,
):
    baseline_accuracy = compute_accuracy_percent(baseline_abs_error, gt_total)
    luv_accuracy = compute_accuracy_percent(luv_abs_error, gt_total)
    improvement = luv_accuracy - baseline_accuracy

    if improvement > 0.0001:
        better_method = "Fused LUV"
    elif improvement < -0.0001:
        better_method = "Grayscale"
    else:
        better_method = "Tie"

    return {
        "Image": image_name,
        "GT Total": gt_total,
        "Baseline Total": baseline_total,
        "Baseline Abs Error": baseline_abs_error,
        "Baseline Accuracy (%)": baseline_accuracy,
        "Fused LUV Total": luv_total,
        "Fused LUV Abs Error": luv_abs_error,
        "Fused LUV Accuracy (%)": luv_accuracy,
        "Improvement (pp)": improvement,
        "Better Method": better_method,
    }


# Function: build_baseline_vs_luv_accuracy_matrix_rows
# Purpose: derive a compact accuracy matrix from the count-ground-truth rows.
def build_baseline_vs_luv_accuracy_matrix_rows(count_ground_truth_rows):
    if len(count_ground_truth_rows) == 0:
        return []

    row_lookup = {}
    ordered_image_names = []

    for row in count_ground_truth_rows:
        row_lookup[(row["Image"], row["Method"])] = row
        if row["Image"] not in ordered_image_names:
            ordered_image_names.append(row["Image"])

    matrix_rows = []
    for image_name in ordered_image_names:
        baseline_row = row_lookup.get((image_name, "Grayscale"))
        luv_row = row_lookup.get((image_name, "Fused LUV"))

        if baseline_row is None or luv_row is None:
            continue

        matrix_rows.append(
            create_accuracy_matrix_row(
                image_name,
                baseline_row["GT Total"],
                baseline_row["Detected Total"],
                baseline_row["Absolute Total Error"],
                luv_row["Detected Total"],
                luv_row["Absolute Total Error"],
            )
        )

    return matrix_rows


# Function: create_overall_method_comparison_row
# Purpose: summarize direct grayscale-vs-LUV agreement across all images.
def create_overall_method_comparison_row(overall_metrics):
    shape_agreement = None
    if overall_metrics["matched_count"] > 0:
        shape_agreement = overall_metrics["shape_match_correct"] / overall_metrics["matched_count"]
    agreement_rate = 1.0
    if overall_metrics["union_total"] > 0:
        agreement_rate = overall_metrics["matched_count"] / overall_metrics["union_total"]

    return {
        "Image": "ALL IMAGES",
        "Baseline Total": overall_metrics["baseline_total"],
        "Fused LUV Total": overall_metrics["fused_total"],
        "Matched Objects": overall_metrics["matched_count"],
        "Baseline Only": overall_metrics["baseline_only"],
        "LUV Only": overall_metrics["luv_only"],
        "Agreement Rate": agreement_rate,
        "Mean Match IoU": safe_divide(overall_metrics["iou_sum"], overall_metrics["matched_count"]),
        "Shape Agreement": shape_agreement,
        "Count Delta": overall_metrics["fused_total"] - overall_metrics["baseline_total"],
    }


# Function: create_overall_channel_match_row
# Purpose: summarize grayscale-vs-single-channel agreement across all images.
def create_overall_channel_match_row(channel_name, overall_metrics):
    shape_agreement = None
    if overall_metrics["matched_count"] > 0:
        shape_agreement = overall_metrics["shape_match_correct"] / overall_metrics["matched_count"]
    agreement_rate = 1.0
    if overall_metrics["union_total"] > 0:
        agreement_rate = overall_metrics["matched_count"] / overall_metrics["union_total"]

    return {
        "Image": "ALL IMAGES",
        "Channel": channel_name,
        "Baseline Total": overall_metrics["baseline_total"],
        "Channel Total": overall_metrics["channel_total"],
        "Matched Objects": overall_metrics["matched_count"],
        "Baseline Only": overall_metrics["baseline_only"],
        "Channel Only": overall_metrics["channel_only"],
        "Agreement Rate": agreement_rate,
        "Mean Match IoU": safe_divide(overall_metrics["iou_sum"], overall_metrics["matched_count"]),
        "Shape Agreement": shape_agreement,
        "Count Delta": overall_metrics["channel_total"] - overall_metrics["baseline_total"],
    }


# Function: format_value
# Purpose: render values cleanly for tables and text reports.
def format_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


# Function: format_table
# Purpose: turn a list of rows into a readable table.
def format_table(rows, columns):
    text_rows = []
    for row in rows:
        text_rows.append([format_value(row.get(column, "")) for column in columns])

    column_widths = []
    for index, column in enumerate(columns):
        width = len(column)
        for row in text_rows:
            width = max(width, len(row[index]))
        column_widths.append(width)

    header = " | ".join(columns[i].ljust(column_widths[i]) for i in range(len(columns)))
    separator = "-+-".join("-" * column_widths[i] for i in range(len(columns)))

    lines = [header, separator]
    for row in text_rows:
        lines.append(" | ".join(row[i].ljust(column_widths[i]) for i in range(len(columns))))

    return "\n".join(lines)


# Function: save_rows_as_csv
# Purpose: save any metrics table as a CSV file.
def save_rows_as_csv(csv_path, rows, columns):
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# Function: save_confusion_matrix_csv
# Purpose: save one confusion matrix as a standalone table CSV.
def save_confusion_matrix_csv(csv_path, row_label_name, confusion_matrix):
    column_labels = list(next(iter(confusion_matrix.values())).keys())
    rows = []

    for row_label, row_values in confusion_matrix.items():
        row = {row_label_name: row_label}
        for column_label in column_labels:
            row[column_label] = row_values[column_label]
        rows.append(row)

    save_rows_as_csv(csv_path, rows, [row_label_name] + column_labels)


# Function: save_text_report
# Purpose: save the printed metrics tables into one report file.
def save_text_report(report_path, sections):
    report_parts = []
    for section_title, section_text in sections:
        report_parts.append(section_title)
        report_parts.append(section_text)

    report_path.write_text("\n\n".join(report_parts) + "\n", encoding="utf-8")


# Function: remove_file_if_exists
# Purpose: avoid stale metric files from older runs.
def remove_file_if_exists(file_path):
    if file_path.exists():
        file_path.unlink()


# Function: clear_directory_files
# Purpose: remove old confusion-matrix tables before writing new ones.
def clear_directory_files(directory_path):
    if directory_path.exists() is False:
        return

    for child_path in directory_path.iterdir():
        if child_path.is_file():
            child_path.unlink()


# Function: save_confusion_matrix_tables
# Purpose: export each confusion matrix into its own spreadsheet-friendly CSV.
def save_confusion_matrix_tables(directory_path, confusion_matrices, row_label_name):
    if len(confusion_matrices) == 0:
        clear_directory_files(directory_path)
        if directory_path.exists():
            try:
                directory_path.rmdir()
            except OSError:
                pass
        return []

    directory_path.mkdir(parents=True, exist_ok=True)
    clear_directory_files(directory_path)

    saved_file_names = []
    for matrix_name, confusion_matrix in confusion_matrices.items():
        csv_file_name = sanitize_filename_part(matrix_name) + ".csv"
        save_confusion_matrix_csv(directory_path / csv_file_name, row_label_name, confusion_matrix)
        saved_file_names.append(csv_file_name)

    return saved_file_names


# Function: get_chart_method_label
# Purpose: shorten method names for chart x-axis labels.
def get_chart_method_label(method_name):
    return METHOD_CHART_LABELS.get(method_name, method_name)


# Function: get_chart_method_color
# Purpose: keep method colors consistent across charts.
def get_chart_method_color(method_name):
    return METHOD_CHART_COLORS.get(method_name, (140, 140, 140))


# Function: get_chart_image_label
# Purpose: shorten image names for chart x-axis labels.
def get_chart_image_label(image_name):
    image_key = normalize_lookup_name(Path(image_name).stem)
    return IMAGE_CHART_LABELS.get(image_key, Path(image_name).stem[:10])


# Function: compute_accuracy_percent
# Purpose: convert an absolute error into a 0-100 accuracy score.
def compute_accuracy_percent(absolute_error, ground_truth_total):
    if ground_truth_total <= 0:
        return 0.0
    return max(0.0, 100.0 * (1.0 - (absolute_error / ground_truth_total)))


# Function: create_chart_canvas
# Purpose: create a white image for drawing charts.
def create_chart_canvas(width, height):
    return np.full((height, width, 3), 255, dtype=np.uint8)


# Function: draw_centered_text
# Purpose: place one line of centered text on a chart.
def draw_centered_text(image, text, center_x, baseline_y, font_scale, color, thickness):
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = int(center_x - (text_size[0] / 2))
    cv2.putText(
        image,
        text,
        (text_x, baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# Function: format_chart_value
# Purpose: keep chart labels compact for either whole numbers or decimals.
def format_chart_value(value, suffix=""):
    if abs(value - round(value)) < 0.05:
        return f"{int(round(value))}{suffix}"
    return f"{value:.1f}{suffix}"


# Function: save_bar_chart
# Purpose: save a single-series percentage bar chart as a PNG.
def save_bar_chart(chart_path, title, labels, values, colors):
    if len(labels) == 0:
        remove_file_if_exists(chart_path)
        return

    width = max(900, 150 * len(labels))
    height = 620
    image = create_chart_canvas(width, height)

    left_margin = 90
    right_margin = 40
    top_margin = 100
    bottom_margin = 120
    plot_left = left_margin
    plot_right = width - right_margin
    plot_top = top_margin
    plot_bottom = height - bottom_margin
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    draw_centered_text(image, title, width // 2, 45, 0.95, (20, 20, 20), 2)

    for tick_value in range(0, 101, 20):
        tick_y = plot_bottom - int((tick_value / 100.0) * plot_height)
        cv2.line(image, (plot_left, tick_y), (plot_right, tick_y), (225, 225, 225), 1)
        cv2.putText(
            image,
            str(tick_value),
            (20, tick_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )

    cv2.line(image, (plot_left, plot_top), (plot_left, plot_bottom), (80, 80, 80), 2)
    cv2.line(image, (plot_left, plot_bottom), (plot_right, plot_bottom), (80, 80, 80), 2)

    group_width = plot_width / len(labels)
    bar_width = int(group_width * 0.55)

    for index, label in enumerate(labels):
        clamped_value = max(0.0, min(100.0, values[index]))
        x_center = int(plot_left + ((index + 0.5) * group_width))
        x1 = int(x_center - (bar_width / 2))
        x2 = int(x_center + (bar_width / 2))
        bar_height = int((clamped_value / 100.0) * plot_height)
        y1 = plot_bottom - bar_height

        cv2.rectangle(image, (x1, y1), (x2, plot_bottom), colors[index], -1)
        cv2.rectangle(image, (x1, y1), (x2, plot_bottom), (50, 50, 50), 1)

        draw_centered_text(
            image,
            f"{clamped_value:.1f}%",
            x_center,
            max(y1 - 12, plot_top + 20),
            0.52,
            (35, 35, 35),
            1,
        )
        draw_centered_text(image, label, x_center, height - 45, 0.62, (35, 35, 35), 1)

    cv2.imwrite(str(chart_path), image)


# Function: save_grouped_value_chart
# Purpose: save a grouped non-negative value chart such as detected object counts.
def save_grouped_value_chart(chart_path, title, category_labels, series_names, series_values, series_colors):
    if len(category_labels) == 0 or len(series_names) == 0:
        remove_file_if_exists(chart_path)
        return

    width = max(980, 170 * len(category_labels))
    height = 660
    image = create_chart_canvas(width, height)

    left_margin = 90
    right_margin = 40
    top_margin = 125
    bottom_margin = 120
    plot_left = left_margin
    plot_right = width - right_margin
    plot_top = top_margin
    plot_bottom = height - bottom_margin
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    draw_centered_text(image, title, width // 2, 45, 0.9, (20, 20, 20), 2)

    legend_x = plot_right - 280
    legend_y = 70
    for series_index, series_name in enumerate(series_names):
        box_y = legend_y + (series_index * 28)
        cv2.rectangle(
            image,
            (legend_x, box_y - 12),
            (legend_x + 16, box_y + 4),
            series_colors[series_index],
            -1,
        )
        cv2.rectangle(image, (legend_x, box_y - 12), (legend_x + 16, box_y + 4), (50, 50, 50), 1)
        cv2.putText(
            image,
            series_name,
            (legend_x + 28, box_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )

    flat_values = [value for series in series_values for value in series]
    max_value = max(flat_values) if len(flat_values) > 0 else 0.0
    max_value = max(1.0, max_value)
    padded_max_value = max_value * 1.15
    tick_step = max(1.0, math.ceil(padded_max_value / 5.0))
    chart_max_value = tick_step * 5.0

    for tick_index in range(6):
        tick_value = tick_index * tick_step
        tick_y = plot_bottom - int((tick_value / chart_max_value) * plot_height)
        cv2.line(image, (plot_left, tick_y), (plot_right, tick_y), (225, 225, 225), 1)
        cv2.putText(
            image,
            format_chart_value(tick_value),
            (20, tick_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )

    cv2.line(image, (plot_left, plot_top), (plot_left, plot_bottom), (80, 80, 80), 2)
    cv2.line(image, (plot_left, plot_bottom), (plot_right, plot_bottom), (80, 80, 80), 2)

    group_width = plot_width / len(category_labels)
    group_inner_width = group_width * 0.72
    gap_width = 8
    bar_width = int((group_inner_width - ((len(series_names) - 1) * gap_width)) / len(series_names))
    bar_width = max(bar_width, 12)

    for category_index, category_label in enumerate(category_labels):
        group_left = int(plot_left + (category_index * group_width) + ((group_width - group_inner_width) / 2))

        for series_index in range(len(series_names)):
            clamped_value = max(0.0, series_values[series_index][category_index])
            x1 = group_left + (series_index * (bar_width + gap_width))
            x2 = x1 + bar_width
            bar_height = int((clamped_value / chart_max_value) * plot_height)
            y1 = plot_bottom - bar_height

            cv2.rectangle(image, (x1, y1), (x2, plot_bottom), series_colors[series_index], -1)
            cv2.rectangle(image, (x1, y1), (x2, plot_bottom), (50, 50, 50), 1)

            draw_centered_text(
                image,
                format_chart_value(clamped_value),
                int((x1 + x2) / 2),
                max(y1 - 10, plot_top + 18),
                0.44,
                (35, 35, 35),
                1,
            )

        draw_centered_text(
            image,
            category_label,
            int(plot_left + ((category_index + 0.5) * group_width)),
            height - 45,
            0.58,
            (35, 35, 35),
            1,
        )

    cv2.imwrite(str(chart_path), image)


# Function: save_grouped_bar_chart
# Purpose: save a grouped percentage bar chart as a PNG.
def save_grouped_bar_chart(chart_path, title, category_labels, series_names, series_values, series_colors):
    if len(category_labels) == 0 or len(series_names) == 0:
        remove_file_if_exists(chart_path)
        return

    width = max(980, 170 * len(category_labels))
    height = 660
    image = create_chart_canvas(width, height)

    left_margin = 90
    right_margin = 40
    top_margin = 125
    bottom_margin = 120
    plot_left = left_margin
    plot_right = width - right_margin
    plot_top = top_margin
    plot_bottom = height - bottom_margin
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    draw_centered_text(image, title, width // 2, 45, 0.9, (20, 20, 20), 2)

    legend_x = plot_right - 260
    legend_y = 70
    for series_index, series_name in enumerate(series_names):
        box_y = legend_y + (series_index * 28)
        cv2.rectangle(
            image,
            (legend_x, box_y - 12),
            (legend_x + 16, box_y + 4),
            series_colors[series_index],
            -1,
        )
        cv2.rectangle(image, (legend_x, box_y - 12), (legend_x + 16, box_y + 4), (50, 50, 50), 1)
        cv2.putText(
            image,
            series_name,
            (legend_x + 28, box_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )

    for tick_value in range(0, 101, 20):
        tick_y = plot_bottom - int((tick_value / 100.0) * plot_height)
        cv2.line(image, (plot_left, tick_y), (plot_right, tick_y), (225, 225, 225), 1)
        cv2.putText(
            image,
            str(tick_value),
            (20, tick_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )

    cv2.line(image, (plot_left, plot_top), (plot_left, plot_bottom), (80, 80, 80), 2)
    cv2.line(image, (plot_left, plot_bottom), (plot_right, plot_bottom), (80, 80, 80), 2)

    group_width = plot_width / len(category_labels)
    group_inner_width = group_width * 0.72
    gap_width = 8
    bar_width = int((group_inner_width - ((len(series_names) - 1) * gap_width)) / len(series_names))
    bar_width = max(bar_width, 12)

    for category_index, category_label in enumerate(category_labels):
        group_left = int(plot_left + (category_index * group_width) + ((group_width - group_inner_width) / 2))

        for series_index in range(len(series_names)):
            clamped_value = max(0.0, min(100.0, series_values[series_index][category_index]))
            x1 = group_left + (series_index * (bar_width + gap_width))
            x2 = x1 + bar_width
            bar_height = int((clamped_value / 100.0) * plot_height)
            y1 = plot_bottom - bar_height

            cv2.rectangle(image, (x1, y1), (x2, plot_bottom), series_colors[series_index], -1)
            cv2.rectangle(image, (x1, y1), (x2, plot_bottom), (50, 50, 50), 1)

            draw_centered_text(
                image,
                f"{clamped_value:.1f}%",
                int((x1 + x2) / 2),
                max(y1 - 10, plot_top + 18),
                0.44,
                (35, 35, 35),
                1,
            )

        draw_centered_text(
            image,
            category_label,
            int(plot_left + ((category_index + 0.5) * group_width)),
            height - 45,
            0.58,
            (35, 35, 35),
            1,
        )

    cv2.imwrite(str(chart_path), image)


# Function: save_signed_bar_chart
# Purpose: save a positive/negative percentage-point improvement chart.
def save_signed_bar_chart(chart_path, title, labels, values, positive_color, negative_color):
    if len(labels) == 0:
        remove_file_if_exists(chart_path)
        return

    width = max(940, 155 * len(labels))
    height = 660
    image = create_chart_canvas(width, height)

    left_margin = 90
    right_margin = 40
    top_margin = 100
    bottom_margin = 120
    plot_left = left_margin
    plot_right = width - right_margin
    plot_top = top_margin
    plot_bottom = height - bottom_margin
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    zero_y = plot_top + int(plot_height / 2)

    draw_centered_text(image, title, width // 2, 45, 0.9, (20, 20, 20), 2)

    max_magnitude = max(abs(value) for value in values)
    max_magnitude = max(5.0, max_magnitude)
    tick_step = max(5.0, math.ceil((max_magnitude / 4.0) / 5.0) * 5.0)
    chart_max_magnitude = tick_step * 4.0

    for tick_value in range(-4, 5):
        scaled_tick_value = tick_value * tick_step
        tick_offset = scaled_tick_value / chart_max_magnitude
        tick_y = zero_y - int(tick_offset * (plot_height / 2))
        cv2.line(image, (plot_left, tick_y), (plot_right, tick_y), (225, 225, 225), 1)
        cv2.putText(
            image,
            format_chart_value(scaled_tick_value, "pp"),
            (10, tick_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )

    cv2.line(image, (plot_left, plot_top), (plot_left, plot_bottom), (80, 80, 80), 2)
    cv2.line(image, (plot_left, zero_y), (plot_right, zero_y), (80, 80, 80), 2)

    group_width = plot_width / len(labels)
    bar_width = int(group_width * 0.55)

    for index, label in enumerate(labels):
        value = values[index]
        x_center = int(plot_left + ((index + 0.5) * group_width))
        x1 = int(x_center - (bar_width / 2))
        x2 = int(x_center + (bar_width / 2))
        y_end = zero_y - int((value / chart_max_magnitude) * (plot_height / 2))
        fill_color = positive_color if value >= 0 else negative_color

        top_y = min(zero_y, y_end)
        bottom_y = max(zero_y, y_end)
        cv2.rectangle(image, (x1, top_y), (x2, bottom_y), fill_color, -1)
        cv2.rectangle(image, (x1, top_y), (x2, bottom_y), (50, 50, 50), 1)

        label_y = top_y - 10 if value >= 0 else bottom_y + 25
        label_y = max(plot_top + 18, min(label_y, plot_bottom - 10))
        draw_centered_text(
            image,
            format_chart_value(value, "pp"),
            x_center,
            label_y,
            0.45,
            (35, 35, 35),
            1,
        )
        draw_centered_text(image, label, x_center, height - 45, 0.58, (35, 35, 35), 1)

    cv2.imwrite(str(chart_path), image)


# Function: format_comparison_table
# Purpose: turn the saved comparison rows into a readable table.
def format_comparison_table(summary_rows):
    columns = ["Image", "Method", "Total Objects", "Triangles", "Rectangles", "Polygons"]
    return format_table(summary_rows, columns)


# Function: format_total_comparison_table
# Purpose: turn the short total-comparison rows into a readable table.
def format_total_comparison_table(total_rows):
    columns = ["Image Index", "Grayscale Total", "Merged LUV Total"]
    return format_table(total_rows, columns)


# Function: format_channel_comparison_table
# Purpose: turn the saved L/u/v rows into a readable table.
def format_channel_comparison_table(channel_rows):
    columns = ["Image", "Channel", "Total Objects", "Triangles", "Rectangles", "Polygons"]
    return format_table(channel_rows, columns)


# Function: format_channel_match_table
# Purpose: turn the saved per-channel match rows into a readable table.
def format_channel_match_table(channel_match_rows):
    columns = [
        "Image",
        "Channel",
        "Baseline Total",
        "Channel Total",
        "Matched Objects",
        "Baseline Only",
        "Channel Only",
        "Agreement Rate",
        "Mean Match IoU",
        "Shape Agreement",
        "Count Delta",
    ]
    return format_table(channel_match_rows, columns)


# Function: format_count_ground_truth_table
# Purpose: turn the saved correct-count rows into a readable table.
def format_count_ground_truth_table(count_ground_truth_rows):
    columns = [
        "Image",
        "Method",
        "GT Total",
        "Detected Total",
        "Total Error",
        "Absolute Total Error",
        "GT Triangles",
        "Detected Triangles",
        "Triangle Error",
        "GT Rectangles",
        "Detected Rectangles",
        "Rectangle Error",
        "GT Polygons",
        "Detected Polygons",
        "Polygon Error",
        "Shape Abs Error",
        "Exact Total Match",
        "Exact Shape Match",
    ]
    return format_table(count_ground_truth_rows, columns)


# Function: format_accuracy_matrix_table
# Purpose: turn the compact baseline-vs-LUV matrix into a readable table.
def format_accuracy_matrix_table(accuracy_matrix_rows):
    columns = [
        "Image",
        "GT Total",
        "Baseline Total",
        "Baseline Abs Error",
        "Baseline Accuracy (%)",
        "Fused LUV Total",
        "Fused LUV Abs Error",
        "Fused LUV Accuracy (%)",
        "Improvement (pp)",
        "Better Method",
    ]
    return format_table(accuracy_matrix_rows, columns)


# Function: format_confusion_matrix
# Purpose: turn one confusion matrix into a readable table.
def format_confusion_matrix(confusion_matrix, row_label_name):
    column_labels = list(next(iter(confusion_matrix.values())).keys())
    rows = []

    for row_label, row_values in confusion_matrix.items():
        row = {row_label_name: row_label}
        for column_label in column_labels:
            row[column_label] = row_values[column_label]
        rows.append(row)

    return format_table(rows, [row_label_name] + column_labels)


# Function: format_confusion_matrix_sections
# Purpose: join multiple confusion matrices into one text block.
def format_confusion_matrix_sections(confusion_matrices, row_label_name):
    sections = []

    for matrix_name, confusion_matrix in confusion_matrices.items():
        sections.append(matrix_name)
        sections.append(format_confusion_matrix(confusion_matrix, row_label_name))

    return "\n\n".join(sections)


# Function: process_one_image
# Purpose: run the full baseline and LUV pipeline for one image.
# Logic:
# 1. read the image
# 2. run the grayscale baseline
# 3. convert the image to LUV
# 4. process L, u, and v separately
# 5. merge the detections
# 6. count the detected shapes
# 7. compute comparison metrics
# 8. save all output images
# 9. return rows for the summary tables
def process_one_image(image_path, image_index, ground_truth_by_image):
    # -------------------- READING IMAGE --------------------
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Skipping unreadable file: {image_path}")
        return {
            "summary_rows": [],
            "channel_rows": [],
            "channel_match_rows": [],
            "count_ground_truth_rows": [],
            "method_confusion_matrices": {},
            "total_row": None,
            "method_comparison_row": None,
            "ground_truth_rows": [],
            "ground_truth_metrics": {},
            "ground_truth_found": False,
            "ground_truth_confusion_matrices": {},
            "count_ground_truth_metrics": {},
            "count_ground_truth_found": False,
        }

    print("\nProcessing image:", image_path.name)

    # -------------------- APPLYING GRAYSCALE BASELINE --------------------
    # This is the grayscale pipeline used for comparison
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    baseline_detections, baseline_blurred, baseline_edged, baseline_closed, baseline_result = process_channel(
        gray,
        image,
        (255, 255, 255),
        "gray",
    )

    # -------------------- APPLYING LUV COLOR SPACE --------------------
    # This is where LUV is applied in the code
    # The image is changed from BGR color space to LUV color space
    luv_image = cv2.cvtColor(image, cv2.COLOR_BGR2Luv)

    # -------------------- SPLITTING LUV CHANNELS --------------------
    # L = lightness
    # u = green to red information
    # v = blue to yellow information
    l_channel, u_channel, v_channel = cv2.split(luv_image)

    # -------------------- ENHANCING LUV CHANNELS --------------------
    # This makes the L, u, and v channels more sensitive before edge detection
    l_enhanced = enhance_luv_channel(l_channel)
    u_enhanced = enhance_luv_channel(u_channel)
    v_enhanced = enhance_luv_channel(v_channel)

    # -------------------- PROCESSING L CHANNEL --------------------
    # This processes only the lightness channel with more sensitive settings
    l_detections, l_blurred, l_edged, l_closed, l_result = process_channel(
        l_enhanced,
        image,
        (255, 0, 0),
        "L",
        5,
        120,
        (5, 5),
    )

    # -------------------- PROCESSING U CHANNEL --------------------
    # This processes the u channel separately with more sensitive settings
    u_detections, u_blurred, u_edged, u_closed, u_result = process_channel(
        u_enhanced,
        image,
        (0, 255, 0),
        "u",
        5,
        120,
        (5, 5),
    )

    # -------------------- PROCESSING V CHANNEL --------------------
    # This processes the v channel separately with more sensitive settings
    v_detections, v_blurred, v_edged, v_closed, v_result = process_channel(
        v_enhanced,
        image,
        (0, 0, 255),
        "v",
        5,
        120,
        (5, 5),
    )

    # -------------------- MERGING UNIQUE OBJECTS --------------------
    # The detections from L, u, and v are combined here
    all_luv_detections = l_detections + u_detections + v_detections
    fused_detections = merge_unique_objects(all_luv_detections)

    fused_result = image.copy()

    # Draw the final merged detections on one result image
    for detection in fused_detections:
        cv2.drawContours(fused_result, [detection["contour"]], -1, (0, 255, 255), 2)
        x, y, w, h = detection["box"]
        cv2.putText(fused_result, detection["shape"], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # -------------------- COUNTING OBJECTS --------------------
    baseline_triangles, baseline_rectangles, baseline_polygons = count_shapes(baseline_detections)
    l_triangles, l_rectangles, l_polygons = count_shapes(l_detections)
    u_triangles, u_rectangles, u_polygons = count_shapes(u_detections)
    v_triangles, v_rectangles, v_polygons = count_shapes(v_detections)
    fused_triangles, fused_rectangles, fused_polygons = count_shapes(fused_detections)

    baseline_total = len(baseline_detections)
    l_total = len(l_detections)
    u_total = len(u_detections)
    v_total = len(v_detections)
    fused_total = len(fused_detections)
    l_comparison_metrics = compare_detection_sets(baseline_detections, l_detections)
    u_comparison_metrics = compare_detection_sets(baseline_detections, u_detections)
    v_comparison_metrics = compare_detection_sets(baseline_detections, v_detections)
    method_comparison_metrics = compare_detection_sets(baseline_detections, fused_detections)
    method_confusion_matrices = {
        "Fused LUV": build_confusion_matrix(baseline_detections, fused_detections, 0.20),
        "L channel": build_confusion_matrix(baseline_detections, l_detections, 0.20),
        "u channel": build_confusion_matrix(baseline_detections, u_detections, 0.20),
        "v channel": build_confusion_matrix(baseline_detections, v_detections, 0.20),
    }

    # -------------------- DISPLAYING RESULTS --------------------
    print("Baseline grayscale count:", baseline_total)
    print("L channel count:", l_total)
    print("u channel count:", u_total)
    print("v channel count:", v_total)
    print("Fused LUV count:", fused_total)

    print("\nBaseline shape count")
    print("Triangles:", baseline_triangles)
    print("Rectangles:", baseline_rectangles)
    print("Objects with more than four edges:", baseline_polygons)

    print("\nFused LUV shape count")
    print("Triangles:", fused_triangles)
    print("Rectangles:", fused_rectangles)
    print("Objects with more than four edges:", fused_polygons)

    print("\nDirect Method Comparison")
    print("Matched objects:", method_comparison_metrics["matched_count"])
    print("Baseline-only detections:", method_comparison_metrics["baseline_only"])
    print("LUV-only detections:", method_comparison_metrics["luv_only"])
    print("Agreement rate:", format_value(method_comparison_metrics["agreement_rate"]))
    print("Mean matched IoU:", format_value(method_comparison_metrics["mean_iou"]))
    print("Shape agreement:", format_value(method_comparison_metrics["shape_agreement"]))

    print("\nPer-Channel Matches vs Grayscale")
    print(
        "L channel:",
        "matches =", l_comparison_metrics["matched_count"],
        "agreement =", format_value(l_comparison_metrics["agreement_rate"]),
        "mean IoU =", format_value(l_comparison_metrics["mean_iou"]),
    )
    print(
        "u channel:",
        "matches =", u_comparison_metrics["matched_count"],
        "agreement =", format_value(u_comparison_metrics["agreement_rate"]),
        "mean IoU =", format_value(u_comparison_metrics["mean_iou"]),
    )
    print(
        "v channel:",
        "matches =", v_comparison_metrics["matched_count"],
        "agreement =", format_value(v_comparison_metrics["agreement_rate"]),
        "mean IoU =", format_value(v_comparison_metrics["mean_iou"]),
    )

    ground_truth_rows = []
    ground_truth_metrics = {}
    ground_truth_confusion_matrices = {}
    ground_truth_detections = get_ground_truth_for_image(image_path, ground_truth_by_image)
    ground_truth_found = len(ground_truth_detections) > 0
    count_ground_truth_rows = []
    count_ground_truth_metrics = {}
    known_count_ground_truth = get_known_count_ground_truth_for_image(image_path)
    count_ground_truth_found = known_count_ground_truth is not None

    if count_ground_truth_found:
        baseline_count_ground_truth_metrics = evaluate_counts_against_ground_truth(
            baseline_total,
            baseline_triangles,
            baseline_rectangles,
            baseline_polygons,
            known_count_ground_truth,
        )
        l_count_ground_truth_metrics = evaluate_counts_against_ground_truth(
            l_total,
            l_triangles,
            l_rectangles,
            l_polygons,
            known_count_ground_truth,
        )
        u_count_ground_truth_metrics = evaluate_counts_against_ground_truth(
            u_total,
            u_triangles,
            u_rectangles,
            u_polygons,
            known_count_ground_truth,
        )
        v_count_ground_truth_metrics = evaluate_counts_against_ground_truth(
            v_total,
            v_triangles,
            v_rectangles,
            v_polygons,
            known_count_ground_truth,
        )
        fused_count_ground_truth_metrics = evaluate_counts_against_ground_truth(
            fused_total,
            fused_triangles,
            fused_rectangles,
            fused_polygons,
            known_count_ground_truth,
        )

        print("\nCorrect Count Comparison")
        print(
            "Correct count:",
            known_count_ground_truth["total"],
            "| triangles:",
            known_count_ground_truth["triangles"],
            "| rectangles:",
            known_count_ground_truth["rectangles"],
            "| polygons:",
            known_count_ground_truth["polygons"],
        )
        print(
            "Grayscale total error:",
            baseline_count_ground_truth_metrics["total_error"],
            "| shape abs error:",
            baseline_count_ground_truth_metrics["shape_abs_error"],
        )
        print(
            "L channel total error:",
            l_count_ground_truth_metrics["total_error"],
            "| shape abs error:",
            l_count_ground_truth_metrics["shape_abs_error"],
        )
        print(
            "u channel total error:",
            u_count_ground_truth_metrics["total_error"],
            "| shape abs error:",
            u_count_ground_truth_metrics["shape_abs_error"],
        )
        print(
            "v channel total error:",
            v_count_ground_truth_metrics["total_error"],
            "| shape abs error:",
            v_count_ground_truth_metrics["shape_abs_error"],
        )
        print(
            "Fused LUV total error:",
            fused_count_ground_truth_metrics["total_error"],
            "| shape abs error:",
            fused_count_ground_truth_metrics["shape_abs_error"],
        )

        count_ground_truth_rows.append(
            create_count_ground_truth_row(image_path.name, "Grayscale", baseline_count_ground_truth_metrics)
        )
        count_ground_truth_rows.append(
            create_count_ground_truth_row(image_path.name, "L channel", l_count_ground_truth_metrics)
        )
        count_ground_truth_rows.append(
            create_count_ground_truth_row(image_path.name, "u channel", u_count_ground_truth_metrics)
        )
        count_ground_truth_rows.append(
            create_count_ground_truth_row(image_path.name, "v channel", v_count_ground_truth_metrics)
        )
        count_ground_truth_rows.append(
            create_count_ground_truth_row(image_path.name, "Fused LUV", fused_count_ground_truth_metrics)
        )

        count_ground_truth_metrics["Grayscale"] = baseline_count_ground_truth_metrics
        count_ground_truth_metrics["L channel"] = l_count_ground_truth_metrics
        count_ground_truth_metrics["u channel"] = u_count_ground_truth_metrics
        count_ground_truth_metrics["v channel"] = v_count_ground_truth_metrics
        count_ground_truth_metrics["Fused LUV"] = fused_count_ground_truth_metrics

    if ground_truth_path is not None:
        if ground_truth_found:
            baseline_ground_truth_metrics = evaluate_against_ground_truth(baseline_detections, ground_truth_detections)
            l_ground_truth_metrics = evaluate_against_ground_truth(l_detections, ground_truth_detections)
            u_ground_truth_metrics = evaluate_against_ground_truth(u_detections, ground_truth_detections)
            v_ground_truth_metrics = evaluate_against_ground_truth(v_detections, ground_truth_detections)
            fused_ground_truth_metrics = evaluate_against_ground_truth(fused_detections, ground_truth_detections)
            ground_truth_confusion_matrices = {
                "Grayscale": build_confusion_matrix(
                    ground_truth_detections,
                    baseline_detections,
                    0.50,
                    include_unknown=True,
                ),
                "L channel": build_confusion_matrix(
                    ground_truth_detections,
                    l_detections,
                    0.50,
                    include_unknown=True,
                ),
                "u channel": build_confusion_matrix(
                    ground_truth_detections,
                    u_detections,
                    0.50,
                    include_unknown=True,
                ),
                "v channel": build_confusion_matrix(
                    ground_truth_detections,
                    v_detections,
                    0.50,
                    include_unknown=True,
                ),
                "Fused LUV": build_confusion_matrix(
                    ground_truth_detections,
                    fused_detections,
                    0.50,
                    include_unknown=True,
                ),
            }

            print("\nGround Truth Comparison")
            print("Ground truth object count:", len(ground_truth_detections))
            print(
                "Grayscale precision / recall / F1:",
                format_value(baseline_ground_truth_metrics["precision"]),
                format_value(baseline_ground_truth_metrics["recall"]),
                format_value(baseline_ground_truth_metrics["f1"]),
            )
            print(
                "Fused LUV precision / recall / F1:",
                format_value(fused_ground_truth_metrics["precision"]),
                format_value(fused_ground_truth_metrics["recall"]),
                format_value(fused_ground_truth_metrics["f1"]),
            )
            print(
                "L channel precision / recall / F1:",
                format_value(l_ground_truth_metrics["precision"]),
                format_value(l_ground_truth_metrics["recall"]),
                format_value(l_ground_truth_metrics["f1"]),
            )
            print(
                "u channel precision / recall / F1:",
                format_value(u_ground_truth_metrics["precision"]),
                format_value(u_ground_truth_metrics["recall"]),
                format_value(u_ground_truth_metrics["f1"]),
            )
            print(
                "v channel precision / recall / F1:",
                format_value(v_ground_truth_metrics["precision"]),
                format_value(v_ground_truth_metrics["recall"]),
                format_value(v_ground_truth_metrics["f1"]),
            )

            ground_truth_rows.append(
                create_ground_truth_row(image_path.name, "Grayscale", baseline_ground_truth_metrics)
            )
            ground_truth_rows.append(
                create_ground_truth_row(image_path.name, "L channel", l_ground_truth_metrics)
            )
            ground_truth_rows.append(
                create_ground_truth_row(image_path.name, "u channel", u_ground_truth_metrics)
            )
            ground_truth_rows.append(
                create_ground_truth_row(image_path.name, "v channel", v_ground_truth_metrics)
            )
            ground_truth_rows.append(
                create_ground_truth_row(image_path.name, "Fused LUV", fused_ground_truth_metrics)
            )

            ground_truth_metrics["Grayscale"] = baseline_ground_truth_metrics
            ground_truth_metrics["L channel"] = l_ground_truth_metrics
            ground_truth_metrics["u channel"] = u_ground_truth_metrics
            ground_truth_metrics["v channel"] = v_ground_truth_metrics
            ground_truth_metrics["Fused LUV"] = fused_ground_truth_metrics
        else:
            print("\nGround truth not found for this image. Skipping supervised metrics.")

    # -------------------- SAVING OUTPUT IMAGES --------------------
    output_folder = image_path.parent / "luv_output" / image_path.stem
    output_folder.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_folder / "0_original.jpg"), image)

    cv2.imwrite(str(output_folder / "1_baseline_gray.jpg"), gray)
    cv2.imwrite(str(output_folder / "2_baseline_blurred.jpg"), baseline_blurred)
    cv2.imwrite(str(output_folder / "3_baseline_edged.jpg"), baseline_edged)
    cv2.imwrite(str(output_folder / "4_baseline_closed.jpg"), baseline_closed)
    cv2.imwrite(str(output_folder / "5_baseline_result.jpg"), baseline_result)

    cv2.imwrite(str(output_folder / "6_l_channel.jpg"), l_channel)
    cv2.imwrite(str(output_folder / "7_l_enhanced.jpg"), l_enhanced)
    cv2.imwrite(str(output_folder / "8_l_result.jpg"), l_result)
    cv2.imwrite(str(output_folder / "9_u_channel.jpg"), u_channel)
    cv2.imwrite(str(output_folder / "10_u_enhanced.jpg"), u_enhanced)
    cv2.imwrite(str(output_folder / "11_u_result.jpg"), u_result)
    cv2.imwrite(str(output_folder / "12_v_channel.jpg"), v_channel)
    cv2.imwrite(str(output_folder / "13_v_enhanced.jpg"), v_enhanced)
    cv2.imwrite(str(output_folder / "14_v_result.jpg"), v_result)

    cv2.imwrite(str(output_folder / "15_l_edged.jpg"), l_edged)
    cv2.imwrite(str(output_folder / "16_u_edged.jpg"), u_edged)
    cv2.imwrite(str(output_folder / "17_v_edged.jpg"), v_edged)

    cv2.imwrite(str(output_folder / "18_l_closed.jpg"), l_closed)
    cv2.imwrite(str(output_folder / "19_u_closed.jpg"), u_closed)
    cv2.imwrite(str(output_folder / "20_v_closed.jpg"), v_closed)

    cv2.imwrite(str(output_folder / "21_fused_luv_result.jpg"), fused_result)

    print("\nSaved output images in:", output_folder)

    grayscale_row = create_summary_row(
        image_path.name,
        "Grayscale",
        baseline_total,
        baseline_triangles,
        baseline_rectangles,
        baseline_polygons,
    )
    luv_row = create_summary_row(
        image_path.name,
        "Fused LUV",
        fused_total,
        fused_triangles,
        fused_rectangles,
        fused_polygons,
    )
    total_row = create_total_comparison_row(
        image_index,
        baseline_total,
        fused_total,
    )
    method_comparison_row = create_method_comparison_row(image_path.name, method_comparison_metrics)
    channel_rows = [
        create_channel_summary_row(
            image_path.name,
            "L",
            l_total,
            l_triangles,
            l_rectangles,
            l_polygons,
        ),
        create_channel_summary_row(
            image_path.name,
            "u",
            u_total,
            u_triangles,
            u_rectangles,
            u_polygons,
        ),
        create_channel_summary_row(
            image_path.name,
            "v",
            v_total,
            v_triangles,
            v_rectangles,
            v_polygons,
        ),
    ]
    channel_match_rows = [
        create_channel_match_row(image_path.name, "L", l_comparison_metrics),
        create_channel_match_row(image_path.name, "u", u_comparison_metrics),
        create_channel_match_row(image_path.name, "v", v_comparison_metrics),
    ]

    return {
        "summary_rows": [grayscale_row, luv_row],
        "channel_rows": channel_rows,
        "channel_match_rows": channel_match_rows,
        "count_ground_truth_rows": count_ground_truth_rows,
        "method_confusion_matrices": method_confusion_matrices,
        "total_row": total_row,
        "method_comparison_row": method_comparison_row,
        "ground_truth_rows": ground_truth_rows,
        "ground_truth_metrics": ground_truth_metrics,
        "ground_truth_found": ground_truth_found,
        "ground_truth_confusion_matrices": ground_truth_confusion_matrices,
        "count_ground_truth_metrics": count_ground_truth_metrics,
        "count_ground_truth_found": count_ground_truth_found,
    }


# -------------------- PROCESSING ALL IMAGES --------------------
# This loop allows the same code to work for one image or a whole folder
ground_truth_by_image = load_ground_truth(ground_truth_path)
summary_rows = []
channel_rows = []
channel_match_rows = []
count_ground_truth_rows = []
method_confusion_matrix_rows = []
ground_truth_confusion_matrix_rows = []
total_rows = []
method_comparison_rows = []
ground_truth_rows = []
missing_ground_truth_images = []
missing_count_ground_truth_images = []
method_confusion_matrix_tables = {}
ground_truth_confusion_matrix_tables = {}
overall_method_comparison_metrics = {
    "baseline_total": 0,
    "fused_total": 0,
    "matched_count": 0,
    "baseline_only": 0,
    "luv_only": 0,
    "union_total": 0,
    "iou_sum": 0.0,
    "shape_match_correct": 0,
}
overall_channel_match_metrics = {}
overall_ground_truth_metrics = {}
overall_count_ground_truth_metrics = {}
overall_method_confusion_matrices = {}
overall_ground_truth_confusion_matrices = {}

for image_index, image_file in enumerate(image_files, start=1):
    image_results = process_one_image(image_file, image_index, ground_truth_by_image)
    summary_rows += image_results["summary_rows"]
    channel_rows += image_results["channel_rows"]
    channel_match_rows += image_results["channel_match_rows"]
    count_ground_truth_rows += image_results["count_ground_truth_rows"]

    if image_results["total_row"] is not None:
        total_rows.append(image_results["total_row"])

    if image_results["method_comparison_row"] is not None:
        method_comparison_rows.append(image_results["method_comparison_row"])
        overall_method_comparison_metrics["baseline_total"] += image_results["method_comparison_row"]["Baseline Total"]
        overall_method_comparison_metrics["fused_total"] += image_results["method_comparison_row"]["Fused LUV Total"]
        overall_method_comparison_metrics["matched_count"] += image_results["method_comparison_row"]["Matched Objects"]
        overall_method_comparison_metrics["baseline_only"] += image_results["method_comparison_row"]["Baseline Only"]
        overall_method_comparison_metrics["luv_only"] += image_results["method_comparison_row"]["LUV Only"]
        overall_method_comparison_metrics["iou_sum"] += (
            image_results["method_comparison_row"]["Mean Match IoU"]
            * image_results["method_comparison_row"]["Matched Objects"]
        )
        overall_method_comparison_metrics["union_total"] += (
            image_results["method_comparison_row"]["Baseline Total"]
            + image_results["method_comparison_row"]["Fused LUV Total"]
            - image_results["method_comparison_row"]["Matched Objects"]
        )
        if image_results["method_comparison_row"]["Shape Agreement"] is not None:
            overall_method_comparison_metrics["shape_match_correct"] += (
                image_results["method_comparison_row"]["Shape Agreement"]
                * image_results["method_comparison_row"]["Matched Objects"]
            )

    for method_name, confusion_matrix in image_results["method_confusion_matrices"].items():
        matrix_name = f"{image_file.name} - Grayscale vs {method_name}"
        method_confusion_matrix_rows += create_confusion_matrix_rows(
            matrix_name,
            "Grayscale Shape",
            confusion_matrix,
        )
        method_confusion_matrix_tables[matrix_name] = confusion_matrix

        if method_name not in overall_method_confusion_matrices:
            overall_method_confusion_matrices[method_name] = create_empty_confusion_matrix()

        accumulate_confusion_matrix(overall_method_confusion_matrices[method_name], confusion_matrix)

    for channel_match_row in image_results["channel_match_rows"]:
        channel_name = channel_match_row["Channel"]
        if channel_name not in overall_channel_match_metrics:
            overall_channel_match_metrics[channel_name] = {
                "baseline_total": 0,
                "channel_total": 0,
                "matched_count": 0,
                "baseline_only": 0,
                "channel_only": 0,
                "union_total": 0,
                "iou_sum": 0.0,
                "shape_match_correct": 0.0,
            }

        overall_channel_match_metrics[channel_name]["baseline_total"] += channel_match_row["Baseline Total"]
        overall_channel_match_metrics[channel_name]["channel_total"] += channel_match_row["Channel Total"]
        overall_channel_match_metrics[channel_name]["matched_count"] += channel_match_row["Matched Objects"]
        overall_channel_match_metrics[channel_name]["baseline_only"] += channel_match_row["Baseline Only"]
        overall_channel_match_metrics[channel_name]["channel_only"] += channel_match_row["Channel Only"]
        overall_channel_match_metrics[channel_name]["union_total"] += (
            channel_match_row["Baseline Total"]
            + channel_match_row["Channel Total"]
            - channel_match_row["Matched Objects"]
        )
        overall_channel_match_metrics[channel_name]["iou_sum"] += (
            channel_match_row["Mean Match IoU"] * channel_match_row["Matched Objects"]
        )
        if channel_match_row["Shape Agreement"] is not None:
            overall_channel_match_metrics[channel_name]["shape_match_correct"] += (
                channel_match_row["Shape Agreement"] * channel_match_row["Matched Objects"]
            )

    ground_truth_rows += image_results["ground_truth_rows"]

    if ground_truth_path is not None and image_results["ground_truth_found"] is False:
        missing_ground_truth_images.append(image_file.name)

    if image_results["count_ground_truth_found"] is False:
        missing_count_ground_truth_images.append(image_file.name)

    for method_name, metrics in image_results["ground_truth_metrics"].items():
        if method_name not in overall_ground_truth_metrics:
            overall_ground_truth_metrics[method_name] = {
                "gt_total": 0,
                "pred_total": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "iou_sum": 0.0,
                "shape_correct": 0,
                "shape_match_count": 0,
                "absolute_count_error_total": 0,
            }

        overall_ground_truth_metrics[method_name]["gt_total"] += metrics["gt_total"]
        overall_ground_truth_metrics[method_name]["pred_total"] += metrics["pred_total"]
        overall_ground_truth_metrics[method_name]["tp"] += metrics["tp"]
        overall_ground_truth_metrics[method_name]["fp"] += metrics["fp"]
        overall_ground_truth_metrics[method_name]["fn"] += metrics["fn"]
        overall_ground_truth_metrics[method_name]["iou_sum"] += metrics["iou_sum"]
        overall_ground_truth_metrics[method_name]["shape_correct"] += metrics["shape_correct"]
        overall_ground_truth_metrics[method_name]["shape_match_count"] += metrics["shape_match_count"]
        overall_ground_truth_metrics[method_name]["absolute_count_error_total"] += metrics["absolute_count_error"]

    for method_name, confusion_matrix in image_results["ground_truth_confusion_matrices"].items():
        matrix_name = f"{image_file.name} - Ground Truth vs {method_name}"
        ground_truth_confusion_matrix_rows += create_confusion_matrix_rows(
            matrix_name,
            "Ground Truth Shape",
            confusion_matrix,
        )
        ground_truth_confusion_matrix_tables[matrix_name] = confusion_matrix

        if method_name not in overall_ground_truth_confusion_matrices:
            overall_ground_truth_confusion_matrices[method_name] = create_empty_confusion_matrix(
                include_unknown=True
            )

        accumulate_confusion_matrix(overall_ground_truth_confusion_matrices[method_name], confusion_matrix)

    for method_name, metrics in image_results["count_ground_truth_metrics"].items():
        if method_name not in overall_count_ground_truth_metrics:
            overall_count_ground_truth_metrics[method_name] = {
                "gt_total": 0,
                "pred_total": 0,
                "total_error": 0,
                "absolute_total_error": 0,
                "gt_triangles": 0,
                "pred_triangles": 0,
                "triangle_error": 0,
                "gt_rectangles": 0,
                "pred_rectangles": 0,
                "rectangle_error": 0,
                "gt_polygons": 0,
                "pred_polygons": 0,
                "polygon_error": 0,
                "shape_abs_error": 0,
                "exact_total_match_count": 0,
                "exact_shape_match_count": 0,
                "image_count": 0,
            }

        overall_count_ground_truth_metrics[method_name]["gt_total"] += metrics["gt_total"]
        overall_count_ground_truth_metrics[method_name]["pred_total"] += metrics["pred_total"]
        overall_count_ground_truth_metrics[method_name]["total_error"] += metrics["total_error"]
        overall_count_ground_truth_metrics[method_name]["absolute_total_error"] += metrics["absolute_total_error"]
        overall_count_ground_truth_metrics[method_name]["gt_triangles"] += metrics["gt_triangles"]
        overall_count_ground_truth_metrics[method_name]["pred_triangles"] += metrics["pred_triangles"]
        overall_count_ground_truth_metrics[method_name]["triangle_error"] += metrics["triangle_error"]
        overall_count_ground_truth_metrics[method_name]["gt_rectangles"] += metrics["gt_rectangles"]
        overall_count_ground_truth_metrics[method_name]["pred_rectangles"] += metrics["pred_rectangles"]
        overall_count_ground_truth_metrics[method_name]["rectangle_error"] += metrics["rectangle_error"]
        overall_count_ground_truth_metrics[method_name]["gt_polygons"] += metrics["gt_polygons"]
        overall_count_ground_truth_metrics[method_name]["pred_polygons"] += metrics["pred_polygons"]
        overall_count_ground_truth_metrics[method_name]["polygon_error"] += metrics["polygon_error"]
        overall_count_ground_truth_metrics[method_name]["shape_abs_error"] += metrics["shape_abs_error"]
        overall_count_ground_truth_metrics[method_name]["image_count"] += 1
        if metrics["exact_total_match"]:
            overall_count_ground_truth_metrics[method_name]["exact_total_match_count"] += 1
        if metrics["exact_shape_match"]:
            overall_count_ground_truth_metrics[method_name]["exact_shape_match_count"] += 1


# -------------------- FINAL COMPARISON TABLE --------------------
# This prints the final comparison between grayscale and fused LUV
if len(summary_rows) > 0:
    comparison_table = format_comparison_table(summary_rows)
    print("\nComparison Table")
    print(comparison_table)

    if len(total_rows) > 0:
        total_comparison_table = format_total_comparison_table(total_rows)
        print("\nTotal Comparison Table")
        print(total_comparison_table)

    report_sections = [("Comparison Table", comparison_table)]
    generated_chart_paths = []

    comparison_columns = ["Image", "Method", "Total Objects", "Triangles", "Rectangles", "Polygons"]
    save_rows_as_csv(metrics_output_folder / "detection_summary.csv", summary_rows, comparison_columns)

    if len(channel_rows) > 0:
        overall_channel_totals = {}
        for row in channel_rows:
            channel_name = row["Channel"]
            if channel_name not in overall_channel_totals:
                overall_channel_totals[channel_name] = {
                    "Image": "ALL IMAGES",
                    "Channel": channel_name,
                    "Total Objects": 0,
                    "Triangles": 0,
                    "Rectangles": 0,
                    "Polygons": 0,
                }

            overall_channel_totals[channel_name]["Total Objects"] += row["Total Objects"]
            overall_channel_totals[channel_name]["Triangles"] += row["Triangles"]
            overall_channel_totals[channel_name]["Rectangles"] += row["Rectangles"]
            overall_channel_totals[channel_name]["Polygons"] += row["Polygons"]

        channel_rows_with_totals = channel_rows + list(overall_channel_totals.values())
        channel_comparison_table = format_channel_comparison_table(channel_rows_with_totals)
        print("\nLUV Channel Table")
        print(channel_comparison_table)
        channel_columns = ["Image", "Channel", "Total Objects", "Triangles", "Rectangles", "Polygons"]
        save_rows_as_csv(metrics_output_folder / "luv_channel_metrics.csv", channel_rows_with_totals, channel_columns)
        report_sections.append(("LUV Channel Table", channel_comparison_table))
    else:
        remove_file_if_exists(metrics_output_folder / "luv_channel_metrics.csv")

    if len(total_rows) > 0:
        total_columns = ["Image Index", "Grayscale Total", "Merged LUV Total"]
        save_rows_as_csv(metrics_output_folder / "total_count_comparison.csv", total_rows, total_columns)
        report_sections.append(("Total Comparison Table", total_comparison_table))
    else:
        remove_file_if_exists(metrics_output_folder / "total_count_comparison.csv")

    if len(method_comparison_rows) > 0:
        overall_method_row = create_overall_method_comparison_row(overall_method_comparison_metrics)
        method_comparison_rows_with_total = method_comparison_rows + [overall_method_row]
        method_columns = [
            "Image",
            "Baseline Total",
            "Fused LUV Total",
            "Matched Objects",
            "Baseline Only",
            "LUV Only",
            "Agreement Rate",
            "Mean Match IoU",
            "Shape Agreement",
            "Count Delta",
        ]
        method_comparison_table = format_table(method_comparison_rows_with_total, method_columns)
        print("\nMethod Agreement Table")
        print(method_comparison_table)
        save_rows_as_csv(
            metrics_output_folder / "method_agreement_metrics.csv",
            method_comparison_rows_with_total,
            method_columns,
        )
        report_sections.append(("Method Agreement Table", method_comparison_table))
    else:
        remove_file_if_exists(metrics_output_folder / "method_agreement_metrics.csv")

    if len(channel_match_rows) > 0:
        overall_channel_match_rows = []
        for channel_name, metrics in overall_channel_match_metrics.items():
            overall_channel_match_rows.append(create_overall_channel_match_row(channel_name, metrics))

        all_channel_match_rows = channel_match_rows + overall_channel_match_rows
        channel_match_columns = [
            "Image",
            "Channel",
            "Baseline Total",
            "Channel Total",
            "Matched Objects",
            "Baseline Only",
            "Channel Only",
            "Agreement Rate",
            "Mean Match IoU",
            "Shape Agreement",
            "Count Delta",
        ]
        channel_match_table = format_channel_match_table(all_channel_match_rows)
        print("\nLUV Channel Match Table")
        print(channel_match_table)
        save_rows_as_csv(
            metrics_output_folder / "luv_channel_match_metrics.csv",
            all_channel_match_rows,
            channel_match_columns,
        )
        report_sections.append(("LUV Channel Match Table", channel_match_table))
    else:
        remove_file_if_exists(metrics_output_folder / "luv_channel_match_metrics.csv")

    if len(method_confusion_matrix_rows) > 0:
        overall_method_confusion_matrix_rows = []
        overall_method_confusion_sections = {}
        for method_name, confusion_matrix in overall_method_confusion_matrices.items():
            matrix_name = f"ALL IMAGES - Grayscale vs {method_name}"
            overall_method_confusion_matrix_rows += create_confusion_matrix_rows(
                matrix_name,
                "Grayscale Shape",
                confusion_matrix,
            )
            overall_method_confusion_sections[matrix_name] = confusion_matrix
            method_confusion_matrix_tables[matrix_name] = confusion_matrix

        all_method_confusion_matrix_rows = method_confusion_matrix_rows + overall_method_confusion_matrix_rows
        method_confusion_matrix_columns = [
            "Matrix",
            "Grayscale Shape",
            "triangle",
            "rectangle",
            "polygon",
            "unmatched",
        ]
        overall_method_confusion_text = format_confusion_matrix_sections(
            overall_method_confusion_sections,
            "Grayscale Shape",
        )
        print("\nOverall Method Confusion Matrices")
        print(overall_method_confusion_text)
        save_rows_as_csv(
            metrics_output_folder / "method_confusion_matrices_long.csv",
            all_method_confusion_matrix_rows,
            method_confusion_matrix_columns,
        )
        remove_file_if_exists(metrics_output_folder / "method_confusion_matrices.csv")
        save_confusion_matrix_tables(
            metrics_output_folder / "method_confusion_matrix_tables",
            method_confusion_matrix_tables,
            "Grayscale Shape",
        )
        report_sections.append(("Overall Method Confusion Matrices", overall_method_confusion_text))
        report_sections.append(
            (
                "Method Confusion Matrix CSV Tables",
                f"Spreadsheet-ready tables saved in: {metrics_output_folder / 'method_confusion_matrix_tables'}",
            )
        )
    else:
        remove_file_if_exists(metrics_output_folder / "method_confusion_matrices_long.csv")
        remove_file_if_exists(metrics_output_folder / "method_confusion_matrices.csv")
        save_confusion_matrix_tables(
            metrics_output_folder / "method_confusion_matrix_tables",
            {},
            "Grayscale Shape",
        )

    if len(count_ground_truth_rows) > 0:
        overall_count_ground_truth_rows = []
        for method_name, metrics in overall_count_ground_truth_metrics.items():
            overall_count_ground_truth_rows.append(create_overall_count_ground_truth_row(method_name, metrics))

        all_count_ground_truth_rows = count_ground_truth_rows + overall_count_ground_truth_rows
        accuracy_matrix_rows = build_baseline_vs_luv_accuracy_matrix_rows(all_count_ground_truth_rows)
        count_ground_truth_columns = [
            "Image",
            "Method",
            "GT Total",
            "Detected Total",
            "Total Error",
            "Absolute Total Error",
            "GT Triangles",
            "Detected Triangles",
            "Triangle Error",
            "GT Rectangles",
            "Detected Rectangles",
            "Rectangle Error",
            "GT Polygons",
            "Detected Polygons",
            "Polygon Error",
            "Shape Abs Error",
            "Exact Total Match",
            "Exact Shape Match",
        ]
        count_ground_truth_table = format_count_ground_truth_table(all_count_ground_truth_rows)
        print("\nCorrect Count Metrics Table")
        print(count_ground_truth_table)
        save_rows_as_csv(
            metrics_output_folder / "count_ground_truth_metrics.csv",
            all_count_ground_truth_rows,
            count_ground_truth_columns,
        )
        report_sections.append(("Correct Count Metrics Table", count_ground_truth_table))

        if len(accuracy_matrix_rows) > 0:
            accuracy_matrix_columns = [
                "Image",
                "GT Total",
                "Baseline Total",
                "Baseline Abs Error",
                "Baseline Accuracy (%)",
                "Fused LUV Total",
                "Fused LUV Abs Error",
                "Fused LUV Accuracy (%)",
                "Improvement (pp)",
                "Better Method",
            ]
            accuracy_matrix_table = format_accuracy_matrix_table(accuracy_matrix_rows)
            print("\nBaseline vs Fused LUV Accuracy Matrix")
            print(accuracy_matrix_table)
            save_rows_as_csv(
                metrics_output_folder / "baseline_vs_luv_accuracy_matrix.csv",
                accuracy_matrix_rows,
                accuracy_matrix_columns,
            )
            report_sections.append(("Baseline vs Fused LUV Accuracy Matrix", accuracy_matrix_table))
        else:
            remove_file_if_exists(metrics_output_folder / "baseline_vs_luv_accuracy_matrix.csv")

        ordered_chart_methods = ["Grayscale", "L channel", "u channel", "v channel", "Fused LUV"]
        available_chart_methods = [
            method_name for method_name in ordered_chart_methods if method_name in overall_count_ground_truth_metrics
        ]
        chart_method_labels = [get_chart_method_label(method_name) for method_name in available_chart_methods]
        chart_method_colors = [get_chart_method_color(method_name) for method_name in available_chart_methods]

        total_count_accuracy_values = [
            compute_accuracy_percent(
                overall_count_ground_truth_metrics[method_name]["absolute_total_error"],
                overall_count_ground_truth_metrics[method_name]["gt_total"],
            )
            for method_name in available_chart_methods
        ]
        total_count_accuracy_chart_path = metrics_output_folder / "total_count_accuracy_by_method.png"
        save_bar_chart(
            total_count_accuracy_chart_path,
            "Overall Total Count Accuracy",
            chart_method_labels,
            total_count_accuracy_values,
            chart_method_colors,
        )
        generated_chart_paths.append(total_count_accuracy_chart_path.name)

        exact_total_match_values = [
            100.0
            * safe_divide(
                overall_count_ground_truth_metrics[method_name]["exact_total_match_count"],
                overall_count_ground_truth_metrics[method_name]["image_count"],
            )
            for method_name in available_chart_methods
        ]
        exact_shape_match_values = [
            100.0
            * safe_divide(
                overall_count_ground_truth_metrics[method_name]["exact_shape_match_count"],
                overall_count_ground_truth_metrics[method_name]["image_count"],
            )
            for method_name in available_chart_methods
        ]
        exact_match_chart_path = metrics_output_folder / "exact_match_rate_by_method.png"
        save_grouped_bar_chart(
            exact_match_chart_path,
            "Exact Match Rate",
            chart_method_labels,
            ["Exact Total", "Exact Shape"],
            [exact_total_match_values, exact_shape_match_values],
            [(70, 130, 180), (150, 90, 200)],
        )
        generated_chart_paths.append(exact_match_chart_path.name)

        per_image_lookup = {}
        ordered_image_names = []
        for row in count_ground_truth_rows:
            per_image_lookup[(row["Image"], row["Method"])] = row
            if row["Image"] not in ordered_image_names:
                ordered_image_names.append(row["Image"])

        per_image_labels = [get_chart_image_label(image_name) for image_name in ordered_image_names]
        per_image_series_values = []
        for method_name in available_chart_methods:
            method_values = []
            for image_name in ordered_image_names:
                row = per_image_lookup[(image_name, method_name)]
                method_values.append(
                    compute_accuracy_percent(row["Absolute Total Error"], row["GT Total"])
                )
            per_image_series_values.append(method_values)

        per_image_chart_path = metrics_output_folder / "count_accuracy_by_image.png"
        save_grouped_bar_chart(
            per_image_chart_path,
            "Per-Image Total Count Accuracy",
            per_image_labels,
            chart_method_labels,
            per_image_series_values,
            chart_method_colors,
        )
        generated_chart_paths.append(per_image_chart_path.name)

        baseline_vs_luv_rows = [
            row for row in accuracy_matrix_rows if row["Image"] != "ALL IMAGES"
        ]
        if len(baseline_vs_luv_rows) > 0:
            baseline_vs_luv_labels = [get_chart_image_label(row["Image"]) for row in baseline_vs_luv_rows]
            baseline_accuracy_values = [row["Baseline Accuracy (%)"] for row in baseline_vs_luv_rows]
            luv_accuracy_values = [row["Fused LUV Accuracy (%)"] for row in baseline_vs_luv_rows]
            improvement_values = [row["Improvement (pp)"] for row in baseline_vs_luv_rows]
            gt_count_values = [row["GT Total"] for row in baseline_vs_luv_rows]
            baseline_count_values = [row["Baseline Total"] for row in baseline_vs_luv_rows]
            luv_count_values = [row["Fused LUV Total"] for row in baseline_vs_luv_rows]

            baseline_vs_luv_accuracy_chart_path = metrics_output_folder / "baseline_vs_luv_accuracy_by_image.png"
            save_grouped_bar_chart(
                baseline_vs_luv_accuracy_chart_path,
                "Baseline vs Fused LUV Accuracy",
                baseline_vs_luv_labels,
                ["Grayscale", "Fused LUV"],
                [baseline_accuracy_values, luv_accuracy_values],
                [get_chart_method_color("Grayscale"), get_chart_method_color("Fused LUV")],
            )
            generated_chart_paths.append(baseline_vs_luv_accuracy_chart_path.name)

            improvement_chart_path = metrics_output_folder / "baseline_vs_luv_accuracy_gain.png"
            save_signed_bar_chart(
                improvement_chart_path,
                "Fused LUV Accuracy Gain over Grayscale",
                baseline_vs_luv_labels,
                improvement_values,
                get_chart_method_color("Fused LUV"),
                (90, 90, 200),
            )
            generated_chart_paths.append(improvement_chart_path.name)

            count_comparison_chart_path = metrics_output_folder / "baseline_vs_luv_detected_counts.png"
            save_grouped_value_chart(
                count_comparison_chart_path,
                "Ground Truth vs Detected Object Counts",
                baseline_vs_luv_labels,
                ["Ground Truth", "Grayscale", "Fused LUV"],
                [gt_count_values, baseline_count_values, luv_count_values],
                [(55, 55, 55), (155, 155, 155), get_chart_method_color("Fused LUV")],
            )
            generated_chart_paths.append(count_comparison_chart_path.name)
        else:
            remove_file_if_exists(metrics_output_folder / "baseline_vs_luv_accuracy_by_image.png")
            remove_file_if_exists(metrics_output_folder / "baseline_vs_luv_accuracy_gain.png")
            remove_file_if_exists(metrics_output_folder / "baseline_vs_luv_detected_counts.png")
    else:
        remove_file_if_exists(metrics_output_folder / "count_ground_truth_metrics.csv")
        remove_file_if_exists(metrics_output_folder / "total_count_accuracy_by_method.png")
        remove_file_if_exists(metrics_output_folder / "exact_match_rate_by_method.png")
        remove_file_if_exists(metrics_output_folder / "count_accuracy_by_image.png")
        remove_file_if_exists(metrics_output_folder / "baseline_vs_luv_accuracy_matrix.csv")
        remove_file_if_exists(metrics_output_folder / "baseline_vs_luv_accuracy_by_image.png")
        remove_file_if_exists(metrics_output_folder / "baseline_vs_luv_accuracy_gain.png")
        remove_file_if_exists(metrics_output_folder / "baseline_vs_luv_detected_counts.png")

    if len(ground_truth_rows) > 0:
        overall_ground_truth_rows = []
        for method_name, metrics in overall_ground_truth_metrics.items():
            overall_ground_truth_rows.append(create_overall_ground_truth_row(method_name, metrics))

        all_ground_truth_rows = ground_truth_rows + overall_ground_truth_rows
        ground_truth_columns = [
            "Image",
            "Method",
            "GT Objects",
            "Detected Objects",
            "TP",
            "FP",
            "FN",
            "Precision",
            "Recall",
            "F1",
            "Mean IoU",
            "Shape Accuracy",
            "Count Error",
            "Absolute Count Error",
        ]
        ground_truth_table = format_table(all_ground_truth_rows, ground_truth_columns)
        print("\nGround Truth Metrics Table")
        print(ground_truth_table)
        save_rows_as_csv(
            metrics_output_folder / "ground_truth_metrics.csv",
            all_ground_truth_rows,
            ground_truth_columns,
        )
        report_sections.append(("Ground Truth Metrics Table", ground_truth_table))

        ordered_chart_methods = ["Grayscale", "L channel", "u channel", "v channel", "Fused LUV"]
        available_chart_methods = [method_name for method_name in ordered_chart_methods if method_name in overall_ground_truth_metrics]
        box_metric_labels = [get_chart_method_label(method_name) for method_name in available_chart_methods]
        precision_values = [
            100.0
            * safe_divide(
                overall_ground_truth_metrics[method_name]["tp"],
                overall_ground_truth_metrics[method_name]["tp"] + overall_ground_truth_metrics[method_name]["fp"],
            )
            for method_name in available_chart_methods
        ]
        recall_values = [
            100.0
            * safe_divide(
                overall_ground_truth_metrics[method_name]["tp"],
                overall_ground_truth_metrics[method_name]["tp"] + overall_ground_truth_metrics[method_name]["fn"],
            )
            for method_name in available_chart_methods
        ]
        f1_values = []
        for method_name in available_chart_methods:
            precision_value = safe_divide(
                overall_ground_truth_metrics[method_name]["tp"],
                overall_ground_truth_metrics[method_name]["tp"] + overall_ground_truth_metrics[method_name]["fp"],
            )
            recall_value = safe_divide(
                overall_ground_truth_metrics[method_name]["tp"],
                overall_ground_truth_metrics[method_name]["tp"] + overall_ground_truth_metrics[method_name]["fn"],
            )
            f1_values.append(100.0 * safe_divide(2 * precision_value * recall_value, precision_value + recall_value))

        box_metric_chart_path = metrics_output_folder / "box_ground_truth_metrics_by_method.png"
        save_grouped_bar_chart(
            box_metric_chart_path,
            "Ground Truth Detection Metrics",
            box_metric_labels,
            ["Precision", "Recall", "F1"],
            [precision_values, recall_values, f1_values],
            [(70, 130, 180), (80, 175, 76), (150, 90, 200)],
        )
        generated_chart_paths.append(box_metric_chart_path.name)
    else:
        remove_file_if_exists(metrics_output_folder / "ground_truth_metrics.csv")
        remove_file_if_exists(metrics_output_folder / "box_ground_truth_metrics_by_method.png")

    if len(ground_truth_confusion_matrix_rows) > 0:
        overall_ground_truth_confusion_matrix_rows = []
        overall_ground_truth_confusion_sections = {}
        for method_name, confusion_matrix in overall_ground_truth_confusion_matrices.items():
            matrix_name = f"ALL IMAGES - Ground Truth vs {method_name}"
            overall_ground_truth_confusion_matrix_rows += create_confusion_matrix_rows(
                matrix_name,
                "Ground Truth Shape",
                confusion_matrix,
            )
            overall_ground_truth_confusion_sections[matrix_name] = confusion_matrix
            ground_truth_confusion_matrix_tables[matrix_name] = confusion_matrix

        all_ground_truth_confusion_matrix_rows = (
            ground_truth_confusion_matrix_rows + overall_ground_truth_confusion_matrix_rows
        )
        ground_truth_confusion_matrix_columns = [
            "Matrix",
            "Ground Truth Shape",
            "triangle",
            "rectangle",
            "polygon",
            "unknown",
            "unmatched",
        ]
        overall_ground_truth_confusion_text = format_confusion_matrix_sections(
            overall_ground_truth_confusion_sections,
            "Ground Truth Shape",
        )
        print("\nOverall Ground Truth Confusion Matrices")
        print(overall_ground_truth_confusion_text)
        save_rows_as_csv(
            metrics_output_folder / "ground_truth_confusion_matrices_long.csv",
            all_ground_truth_confusion_matrix_rows,
            ground_truth_confusion_matrix_columns,
        )
        remove_file_if_exists(metrics_output_folder / "ground_truth_confusion_matrices.csv")
        save_confusion_matrix_tables(
            metrics_output_folder / "ground_truth_confusion_matrix_tables",
            ground_truth_confusion_matrix_tables,
            "Ground Truth Shape",
        )
        report_sections.append(("Overall Ground Truth Confusion Matrices", overall_ground_truth_confusion_text))
        report_sections.append(
            (
                "Ground Truth Confusion Matrix CSV Tables",
                f"Spreadsheet-ready tables saved in: {metrics_output_folder / 'ground_truth_confusion_matrix_tables'}",
            )
        )
    else:
        remove_file_if_exists(metrics_output_folder / "ground_truth_confusion_matrices_long.csv")
        remove_file_if_exists(metrics_output_folder / "ground_truth_confusion_matrices.csv")
        save_confusion_matrix_tables(
            metrics_output_folder / "ground_truth_confusion_matrix_tables",
            {},
            "Ground Truth Shape",
        )

    if len(missing_count_ground_truth_images) > 0 and len(missing_count_ground_truth_images) < len(image_files):
        missing_count_ground_truth_text = "\n".join(missing_count_ground_truth_images)
        print("\nImages Without Correct Count Ground Truth")
        print(missing_count_ground_truth_text)
        report_sections.append(("Images Without Correct Count Ground Truth", missing_count_ground_truth_text))

    if len(missing_ground_truth_images) > 0:
        missing_ground_truth_text = "\n".join(missing_ground_truth_images)
        print("\nImages Without Ground Truth")
        print(missing_ground_truth_text)
        report_sections.append(("Images Without Ground Truth", missing_ground_truth_text))

    if len(generated_chart_paths) > 0:
        chart_list_text = "\n".join(generated_chart_paths)
        print("\nSaved Accuracy Charts")
        print(chart_list_text)
        report_sections.append(("Saved Accuracy Charts", chart_list_text))

    save_text_report(metrics_output_folder / "metrics_report.txt", report_sections)
    print("\nSaved metrics in:", metrics_output_folder)
