import cv2
import numpy as np

def draw_lane_lines(image_np, left_lines, right_lines, save_path):
    image_drawn_lane = image_np.copy()  # Python is call-by-reference. 

    for x1, y1, x2, y2 in left_lines:
        cv2.line(image_drawn_lane, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)  # cv2.LINE_AA for anti-aliased lines

    for x1, y1, x2, y2 in right_lines:
        cv2.line(image_drawn_lane, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite(save_path, cv2.cvtColor(image_drawn_lane, cv2.COLOR_RGB2BGR))
    return image_drawn_lane

def create_overlay(image_np, pred_mask, alpha, save_path):
    overlay = image_np.astype(np.float32).copy()  

    # Alpha blending. We only apply the red color to the road area, and keep the non-road area unchanged.
    overlay[pred_mask == 1] = (
        alpha * np.array([255, 0, 0]) +
        (1 - alpha) * overlay[pred_mask == 1]
    )
    
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)  # Ensure pixel values are valid after blending
    
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return overlay