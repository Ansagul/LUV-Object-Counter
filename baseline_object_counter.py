#!/usr/bin/env python3

# -------------------- IMPORTING LIBRARIES --------------------
from pathlib import Path
import cv2


# -------------------- TAKING FOLDER NAME FROM USER --------------------
folder_name = input("Enter the folder path that contains the pictures: ")
folder_path = Path(folder_name)


# -------------------- READING IMAGES FROM FOLDER --------------------
image_files = []
image_files += sorted(folder_path.glob("*.jpg"))
image_files += sorted(folder_path.glob("*.jpeg"))
image_files += sorted(folder_path.glob("*.png"))
image_files += sorted(folder_path.glob("*.bmp"))


# -------------------- SAVING OUTPUT IMAGES --------------------
# A new folder is created to save all output images
main_output_folder = folder_path / "baseline_output"
main_output_folder.mkdir(exist_ok=True)


# -------------------- PROCESSING EACH IMAGE --------------------
for image_file in image_files:
    print("\nProcessing image:", image_file.name)

    # -------------------- READING IMAGE --------------------
    image = cv2.imread(str(image_file))

    # -------------------- APPLYING GRAYSCALE --------------------
    # This changes the image from color to black and white
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # -------------------- BLURRING THE IMAGE --------------------
    # This reduces noise in the image
    # Paper parameter: GaussianBlur(gray, (3, 3), 0)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # -------------------- FINDING EDGES --------------------
    # This finds the boundaries of the objects
    # Paper parameter: Canny(gray, 10, 250)
    edged = cv2.Canny(blurred, 10, 250)

    # -------------------- CLOSING GAPS --------------------
    # This joins broken edges so the shapes can be detected more clearly
    # Paper parameter: 7x7 kernel with morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # -------------------- FINDING CONTOURS --------------------
    # Contours are the outlines of the objects
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # -------------------- DEFINING SHAPES --------------------
    triangles = 0
    rectangles = 0
    more_than_four_edges = 0

    output = image.copy()

    # -------------------- CLASSIFYING SHAPES --------------------
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)

        # Paper parameter: 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 3:
            triangles += 1
            color = (0, 255, 0)
            name = "triangle"
        elif len(approx) == 4:
            rectangles += 1
            color = (255, 0, 0)
            name = "rectangle"
        elif len(approx) > 4:
            more_than_four_edges += 1
            color = (0, 0, 255)
            name = "polygon"
        else:
            continue

        cv2.drawContours(output, [approx], -1, color, 2)
        x, y, w, h = cv2.boundingRect(approx)
        cv2.putText(output, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # -------------------- COUNTING OBJECTS --------------------
    total_objects = triangles + rectangles + more_than_four_edges

    # -------------------- DISPLAYING RESULTS --------------------
    if triangles == 0:
        print("There isn't any triangular object in the image.")
    elif triangles == 1:
        print("There is 1 triangular object in the image.")
    else:
        print("There are", triangles, "triangular objects in the image.")

    if rectangles == 0:
        print("There isn't any rectangular object in the image.")
    elif rectangles == 1:
        print("There is 1 rectangular object in the image.")
    else:
        print("There are", rectangles, "rectangular objects in the image.")

    if more_than_four_edges == 0:
        print("There isn't any object with more than four edges in the image.")
    elif more_than_four_edges == 1:
        print("There is 1 object with more than four edges in the image.")
    else:
        print("There are", more_than_four_edges, "objects with more than four edges in the image.")

    if total_objects == 1:
        print("I am glad to let you know that there is 1 object in this image.")
    else:
        print("I am glad to let you know that there are", total_objects, "objects in this image.")

    # -------------------- SAVING OUTPUT IMAGES --------------------
    image_output_folder = main_output_folder / image_file.stem
    image_output_folder.mkdir(exist_ok=True)

    cv2.imwrite(str(image_output_folder / "1_gray.jpg"), gray)
    cv2.imwrite(str(image_output_folder / "2_blurred.jpg"), blurred)
    cv2.imwrite(str(image_output_folder / "3_edged.jpg"), edged)
    cv2.imwrite(str(image_output_folder / "4_closed.jpg"), closed)
    cv2.imwrite(str(image_output_folder / "5_result.jpg"), output)

    print("Saved output images in:", image_output_folder)
