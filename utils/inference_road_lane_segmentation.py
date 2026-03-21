from libs.model.restnet101 import build_inference_model
from libs.inference.road_segmentation import predict_road
from libs.inference.road_segmentation import apply_road_mask
from libs.inference.lane_segmentation import detect_lines_with_elsed
from libs.inference.lane_segmentation import split_left_right_lines
from libs.visualization.lane_visualization import draw_lane_lines
from libs.visualization.lane_visualization import create_overlay
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

device = config["model"]["device"]
model_path = config["model"]["model_path"]
image_path = config["input"]["image_path"]
resize_size = tuple(config["input"]["resize_size"])
threshold = config["road_segmentation"]["threshold"]
min_slope = config["lane_segmentation"]["min_slope"]
alpha = config["visualization"]["alpha"]
save_path = config["visualization"]["save_path"]

def main():
    model = build_inference_model(device, model_path)

    resized_image, pred_mask = predict_road(model, image_path, device, resize_size, threshold)
    masked_road = apply_road_mask(resized_image, pred_mask)

    segments, _ = detect_lines_with_elsed(masked_road)
    left_lines, right_lines = split_left_right_lines(segments, resized_image.width, min_slope)

    draw_lane_save_path = save_path.replace(".png", "_lanes.png")
    draw_lane_lines(resized_image, left_lines, right_lines, draw_lane_save_path)

    overlay_save_path = save_path.replace(".png", "_overlay.png")
    create_overlay(resized_image, pred_mask, alpha, overlay_save_path)

if __name__ == "__main__":
    main()