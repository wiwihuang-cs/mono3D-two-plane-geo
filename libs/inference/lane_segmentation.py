import cv2
import pyelsed
import numpy as np

def detect_lines_with_elsed(masked_road):
    gray = cv2.cvtColor(masked_road, cv2.COLOR_RGB2GRAY)  # The masked_road image must come from RGB format. 
    segments, _ = pyelsed.detect(gray)  # segments is a list of line, size is (N, 4), scores is a list of confidence score for each line, size is (N,)
    
    return segments

def split_left_right_lines(segments, image_width, min_slope):
    left_lines = []
    right_lines = []

    center_x = image_width / 2

    for seg in segments.astype(np.int32):
        x1, y1, x2, y2 = seg

        if x2 == x1:  # Avoid division by zero for vertical lines
            continue

        slope = (y2 - y1) / (x2 - x1)
        mid_x = (x1 + x2) / 2

        if abs(slope) < min_slope:  # Filter out lines that are too horizontal.
            continue
        
        if slope < 0 and mid_x < center_x:  # The coordinates of cv are not the same as the math, the left lane has a negative slope. The mid_x is used to determine if the line is on the left or right side of the image.
            left_lines.append((x1, y1, x2, y2))
        elif slope > 0 and mid_x > center_x:  # The right lane has a positive slope
            right_lines.append((x1, y1, x2, y2))

    return left_lines, right_lines