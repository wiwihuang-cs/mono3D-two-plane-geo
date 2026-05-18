"""
Inference pipeline with dynamic positive/negative angle lane-segmentation
dispatch driven by knee-point detection.
Stages
------
1. Road segmentation         (PIDNet)
2. ELSED line detection      (shared)
3. Loose left/right split    (sign + min_slope only, no inner/outer choice yet)
4. Knee detection            → picks positive_angle / negative_angle / either
5. Apply chosen module's     split_left_right_lines  (full inner/outer logic)
6. Piecewise linear fit
7. Pitch estimation

Stage 4 uses an optional temporal hysteresis voter; pass ``hysteresis`` (a
``HysteresisVoter`` instance) to enable it across consecutive frames. For
single-image inference this can be left as None.
"""

import cv2
import numpy as np
import pyelsed

from libs.inference.road_segmentation  import predict_road, apply_road_mask
from libs.inference.lane_segmentation_positive_angle import (
    split_left_right_lines as split_positive,
)
from libs.inference.lane_segmentation_negative_angle import (
    split_left_right_lines as split_negative,
)
from libs.inference.lane_fitting      import (
    collect_points_from_segments, piecewise_linear_fit, compute_lane_widths,
)
from libs.inference.pitch_estimation  import estimate_pitch_from_widths
from libs.inference.knee_detection    import detect_knee, HysteresisVoter, loose_split


# ---------------------------------------------------------------------------
# Shared helpers (duplicated from lane_segmentation_*.py so this file is the
# only place that needs to change when adding more variants)
# ---------------------------------------------------------------------------

def _elsed_detect(masked_road, min_length_near, min_length_far):
    gray = cv2.cvtColor(masked_road, cv2.COLOR_RGB2GRAY)
    segments, _ = pyelsed.detect(gray)
    if len(segments) == 0:
        return segments
    x1, y1, x2, y2 = segments[:, 0], segments[:, 1], segments[:, 2], segments[:, 3]
    lengths = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    mid_y = (y1 + y2) / 2
    H = masked_road.shape[0]
    thr = min_length_far + (min_length_near - min_length_far) * (mid_y / H)
    return segments[lengths >= thr]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def infer_one(
    model, image_path, device, resize_size,
    min_slope, min_segment_length_near, min_segment_length_far, lane_band_tolerance,
    extra_points_per_segment, num_bands, num_samples,
    f_x, f_y, w_real,
    *,
    default_version: str = "positive_angle",
    hysteresis: HysteresisVoter | None = None,
    return_debug: bool = False,
):
    """Run the full pipeline on a single image.

    Parameters
    ----------
    default_version
        Used when knee detection returns ``either`` and no hysteresis history
        is available yet.
    hysteresis
        Optional ``HysteresisVoter`` for temporal smoothing across frames.
        Pass ``None`` for single-frame use.
    return_debug
        If True, also return a dict with intermediate values (chosen version,
        knee result, segments).
    """
    # 1. road segmentation
    resized_image, pred_mask = predict_road(model, image_path, device, resize_size)
    masked_road = apply_road_mask(resized_image, pred_mask)

    # 2. ELSED line detection
    segments = _elsed_detect(masked_road, min_segment_length_near, min_segment_length_far)

    # 3 + 4. knee-based version decision.
    # Always run detect_knee (even with empty inputs → returns 'either' with
    # reason='both_sides_weak') AND always call hysteresis.update so the
    # voter's temporal history is advanced on every frame. Skipping these
    # caused per-frame voter state to diverge from per_frame.csv-derived
    # post-hoc hysteresis when frames had zero ELSED segments.
    if len(segments) == 0:
        left_raw, right_raw = [], []
    else:
        left_raw, right_raw = loose_split(
            segments, resized_image.width, resized_image.height, min_slope)
    knee_result = detect_knee(left_raw, right_raw)
    if hysteresis is not None:
        chosen_version = hysteresis.update(knee_result)
    else:
        chosen_version = (knee_result.version
                          if knee_result.version != "either"
                          else default_version)

    # 5. apply the chosen module's full split
    split_fn = split_positive if chosen_version == "positive_angle" else split_negative
    inner_left, inner_right = split_fn(
        segments, resized_image.width, min_slope,
        resized_image.height, lane_band_tolerance,
    )

    # 6. lane fitting
    left_points  = collect_points_from_segments(inner_left,  extra_points_per_segment)
    right_points = collect_points_from_segments(inner_right, extra_points_per_segment)
    left_fits    = piecewise_linear_fit(left_points,  num_bands)
    right_fits   = piecewise_linear_fit(right_points, num_bands)

    # 7. pitch estimation
    widths = compute_lane_widths(left_fits, right_fits, num_samples)
    pitch  = estimate_pitch_from_widths(widths, f_x, f_y, resized_image.height, w_real)

    if return_debug:
        return pitch, {
            "version": chosen_version,
            "knee_result": knee_result,
            "n_segments": int(len(segments)),
        }
    return pitch
