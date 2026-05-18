import time
import yaml
# from libs.model.resnet101 import build_inference_model
from libs.inference.road_segmentation import load_pidnet, predict_road, apply_road_mask
from libs.inference.lane_segmentation_positive_angle import detect_lines_with_elsed, split_left_right_lines
from libs.visualization.lane_visualization import draw_lane_lines, create_overlay, draw_line_segments
# from libs.inference.lane_segmentation import cluster_left_right, get_best_seed
# from libs.visualization.lane_visualization import draw_kmeans_clusters, draw_lane_seed
from libs.inference.lane_fitting import collect_points_from_segments, piecewise_linear_fit, compute_lane_widths
# from libs.inference.lane_fitting import get_x_at_y
from libs.visualization.lane_visualization import draw_piecewise_fits
from libs.inference.pitch_estimation import estimate_pitch_from_widths

with open("config/inference_road_lane_segmentation.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

device = config["model"]["device"]
model_name = config["model"]["model_name"]
weight_path = config["model"]["weight_path"]
image_path = config["input"]["image_path"]
resize_size = tuple(config["input"]["resize_size"])
# threshold = config["road_segmentation"]["threshold"]
# mask_erosion_kernel = config["road_segmentation"]["mask_erosion_kernel"]
min_segment_length_near = config["line_segmentation"]["min_segment_length_near"]
min_segment_length_far  = config["line_segmentation"]["min_segment_length_far"]
min_slope = config["lane_segmentation"]["min_slope"]
# roi = config["lane_segmentation"]["roi"]
lane_band_tolerance = config["lane_segmentation"]["lane_band_tolerance"]
alpha = config["visualization"]["alpha"]
save_path = config["visualization"]["save_path"]
num_bands = config["lane_fitting"]["num_bands"]
num_samples = config["lane_fitting"]["num_samples"]
extra_points_per_segment = config["lane_fitting"]["extra_points_per_segment"]
# ransac_residual_threshold = config["lane_fitting"]["ransac_residual_threshold"]
f_x = config["pitch_estimation"]["f_x"]
f_y = config["pitch_estimation"]["f_y"]
w_real = config["pitch_estimation"]["w_real"]

def main():
    model = load_pidnet(model_name, weight_path, device)
    t0 = time.perf_counter()

    resized_image, pred_mask = predict_road(model, image_path, device, resize_size)
    masked_road = apply_road_mask(resized_image, pred_mask)
    t1 = time.perf_counter()

    segments = detect_lines_with_elsed(masked_road, min_segment_length_near, min_segment_length_far)
    t2 = time.perf_counter()

    # inner_left, inner_right = split_left_right_lines(segments, resized_image.width, min_slope, resized_image.height, lane_band_tolerance, resized_image)
    # left_seed = get_best_seed(left_lines, True, resized_image.height)
    # right_seed = get_best_seed(right_lines, False, resized_image.height)
    # left_clus, right_clus = cluster_left_right(left_lines, right_lines)
    inner_left, inner_right = split_left_right_lines(
        segments, resized_image.width, min_slope, resized_image.height,
        lane_band_tolerance
    )
    t3 = time.perf_counter()

    left_points = collect_points_from_segments(inner_left, extra_points_per_segment)
    right_points = collect_points_from_segments(inner_right, extra_points_per_segment)
    left_fits = piecewise_linear_fit(left_points, num_bands)
    right_fits = piecewise_linear_fit(right_points, num_bands)
    widths = compute_lane_widths(left_fits, right_fits, num_samples)
    t4 = time.perf_counter()

    pitch = estimate_pitch_from_widths(widths, f_x, f_y, resized_image.height, w_real)
    print(pitch)
    t5 = time.perf_counter()

    print(f"road segmentation:   {(t1-t0)*1000:.1f} ms")
    print(f"line segmentation:      {(t2-t1)*1000:.1f} ms")
    print(f"lane segmentation:   {(t3-t2)*1000:.1f} ms")
    print(f"lane fitting:        {(t4-t3)*1000:.1f} ms")
    print(f"pitch estimation:    {(t5-t4)*1000:.1f} ms")

    overlay_save_path = save_path.replace(".png", "_overlay.png")
    create_overlay(resized_image, pred_mask, alpha, overlay_save_path)
    draw_line_save_path = save_path.replace(".png", "_line_segments.png")
    draw_line_segments(resized_image, segments, draw_line_save_path)
    draw_lane_save_path = save_path.replace(".png", "_lanes.png")
    draw_lane_lines(resized_image, inner_left, inner_right, draw_lane_save_path)
    # draw_lane_seed_save_path = save_path.replace(".png", "_lane_seeds.png")
    # draw_lane_seed(resized_image, left_seed, right_seed, draw_lane_seed_save_path)
    # draw_clus_save_path = save_path.replace(".png", "_lane_clusters.png")
    # draw_kmeans_clusters(resized_image, left_clus, right_clus, draw_clus_save_path)
    draw_piecewise_fits_save_path = save_path.replace(".png", "_lane_fits.png")
    draw_piecewise_fits(resized_image, left_fits, right_fits, widths, draw_piecewise_fits_save_path)
if __name__ == "__main__":
    main()
