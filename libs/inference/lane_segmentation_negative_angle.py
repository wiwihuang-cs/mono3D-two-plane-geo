import numpy as np

"""
1. Split the line segments into left and right lane candidates based on their slope and position in the image.
And finally sort the line segments based on their x-coordinate at the bottom of the image
But in down hile case, we change to find the outermost lane segments
2. Add the adaptive ROI filtering based on mid_y and use ROI filtering based on mid_x instead of x_at_bottom, which is more robust to lane curvature and perspective distortion.
"""

def split_left_right_lines(segments, image_width, min_slope, img_height, lane_band_tolerance, roi_near=0.3, roi_far=0.8):

    left_segments = []
    right_segments = []

    center_x = image_width / 2

    for seg in segments.astype(np.int32):
        x1, y1, x2, y2 = seg

        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        if abs(slope) < min_slope:
            continue

        x_at_bottom = x1 + (img_height - y1) / slope

        if slope < 0 and mid_x < center_x:
            left_segments.append({
                "seg": (x1, y1, x2, y2),
                "x_at_bottom": x_at_bottom,
                "mid_x": mid_x,
                "mid_y": mid_y,
            })
        elif slope > 0 and mid_x > center_x:
            right_segments.append({
                "seg": (x1, y1, x2, y2),
                "x_at_bottom": x_at_bottom,
                "mid_x": mid_x,
                "mid_y": mid_y,
            })

    # Apply adaptive ROI filtering based on mid_y and mid_x
    left_segments  = [l for l in left_segments
                      if center_x * (roi_far + (roi_near - roi_far) * (l["mid_y"] / img_height))
                         <= l["mid_x"] <= center_x]
    right_segments = [l for l in right_segments
                      if center_x <= l["mid_x"] <=
                         center_x * (2 - (roi_far + (roi_near - roi_far) * (l["mid_y"] / img_height)))]

    # Get the outermost lane segments
    left_segments.sort(key=lambda x: x["x_at_bottom"])                  
    right_segments.sort(key=lambda x: x["x_at_bottom"], reverse=True)   

    outermost_left_x  = left_segments[0]["x_at_bottom"]
    outermost_right_x = right_segments[0]["x_at_bottom"]

    inner_left = [l["seg"] for l in left_segments
                  if l["x_at_bottom"] <= (outermost_left_x + lane_band_tolerance)]

    inner_right = [l["seg"] for l in right_segments
                   if l["x_at_bottom"] >= (outermost_right_x - lane_band_tolerance)]

    return inner_left, inner_right