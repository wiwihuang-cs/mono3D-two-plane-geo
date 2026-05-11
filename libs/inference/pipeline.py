from libs.inference.road_segmentation  import predict_road, apply_road_mask
from libs.inference.lane_segmentation_up_hile  import detect_lines_with_elsed, split_left_right_lines
from libs.inference.lane_fitting       import collect_points_from_segments, piecewise_linear_fit, compute_lane_widths
from libs.inference.pitch_estimation   import estimate_pitch_from_widths


def infer_one(
    model, image_path, device, resize_size,
    min_slope, min_segment_length_near, min_segment_length_far, lane_band_tolerance,
    extra_points_per_segment, num_bands, num_samples,
    f_x, f_y, w_real,
):
    # road segmentation
    resized_image, pred_mask = predict_road(model, image_path, device, resize_size)
    masked_road = apply_road_mask(resized_image, pred_mask)

    # line segmentation
    segments = detect_lines_with_elsed(masked_road, min_segment_length_near, min_segment_length_far)

    # lane segmentation
    inner_left, inner_right = split_left_right_lines(
        segments, resized_image.width, min_slope,
        resized_image.height, lane_band_tolerance,
    )

    # lane fitting
    left_points  = collect_points_from_segments(inner_left,  extra_points_per_segment)
    right_points = collect_points_from_segments(inner_right, extra_points_per_segment)
    left_fits    = piecewise_linear_fit(left_points,  num_bands)
    right_fits   = piecewise_linear_fit(right_points, num_bands)
    
    # pitch estimation
    widths       = compute_lane_widths(left_fits, right_fits, num_samples)
    pitch = estimate_pitch_from_widths(widths, f_x, f_y, resized_image.height, w_real)
    return pitch
