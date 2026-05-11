"""
carla_module/realtime_test.py
即時 CARLA 俯仰角估計測試腳本

使用方式：
    python carla_module/realtime_test.py [--host HOST] [--port PORT] [--config-path PATH] [--timeout SEC]

注意：執行前請確認 CARLA 伺服器已啟動，且已安裝 carla Python 套件（版本需與伺服器一致）。
"""

from utils.env_setup import setup_env
setup_env()

import argparse
import queue
import sys
import time

import carla
import cv2
import numpy as np
import torch
import yaml
from PIL import Image

# 確保專案根目錄在 import 路徑中
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from libs.inference.road_segmentation import load_pidnet, apply_road_mask
from libs.inference.lane_segmentation_up_hile import detect_lines_with_elsed, split_left_right_lines
from libs.inference.lane_fitting import (
    collect_points_from_segments,
    piecewise_linear_fit,
    compute_lane_widths,
)
from libs.inference.pitch_estimation import estimate_pitch_from_widths
from carla_module.carla_road_segmentation import predict_road_from_pil
from carla_module.carla_visualization import render_piecewise_fits_to_array


# ---------------------------------------------------------------------------
# 全域影像佇列（maxsize=1：主執行緒仍在推斷時直接丟棄舊幀）
# ---------------------------------------------------------------------------
_frame_queue: queue.Queue = queue.Queue(maxsize=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CARLA 即時俯仰角估計")
    parser.add_argument("--host",        default="127.0.0.1",
                        help="CARLA 伺服器位址（預設：127.0.0.1）")
    parser.add_argument("--port",        type=int, default=2000,
                        help="CARLA 伺服器埠號（預設：2000）")
    parser.add_argument("--config-path", default="config/inference_road_lane_segmentation.yaml",
                        help="推斷設定檔路徑（預設：config/inference_road_lane_segmentation.yaml）")
    parser.add_argument("--timeout",     type=float, default=20.0,
                        help="CARLA 連線逾時秒數（預設：20）")
    parser.add_argument("--map",         default="Town03",
                        help="CARLA 地圖名稱（預設：Town03）")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # CARLA 使用方形像素，FOV=90°、寬度=1024 → f_x = f_y = 512
    # config 中的 f_y=455 來自真實相機的非方形像素，此處覆蓋以符合 CARLA 相機模型
    cfg["pitch_estimation"]["f_y"] = cfg["pitch_estimation"]["f_x"]
    return cfg


def select_device(cfg_device: str) -> torch.device:
    # 優先使用 GPU；若無可用 GPU 則退回 config 中指定的裝置
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device(cfg_device)


def setup_carla(
    host: str, port: int, timeout: float, map_name: str = "Town03"
) -> tuple[carla.Client, carla.World, carla.Vehicle, carla.Sensor]:
    # 連接 CARLA 伺服器
    client = carla.Client(host, port)
    client.set_timeout(timeout)

    # 切換至指定地圖（若已在該地圖則 CARLA 會快速跳過載入）
    print(f"[初始化] 載入地圖：{map_name} ...")
    world = client.load_world(map_name)

    bp_lib = world.get_blueprint_library()

    # 生成 Tesla Model 3 並啟用自動駕駛
    vehicle_bp = bp_lib.find("vehicle.tesla.model3")
    spawn_points = world.get_map().get_spawn_points()
    vehicle: carla.Vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    vehicle.set_autopilot(True)

    # 設定 RGB 相機
    # image_size_x=1024, image_size_y=512, fov=90°
    # → f_x = (1024/2) / tan(45°) = 512，與 config 的 f_x 一致
    camera_bp = bp_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "1024")
    camera_bp.set_attribute("image_size_y", "512")
    camera_bp.set_attribute("fov", "90")

    # 掛載於車頂前方
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera: carla.Sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    return client, world, vehicle, camera


def _camera_callback(image: carla.Image) -> None:
    # CARLA 在獨立執行緒觸發此 callback；僅做格式轉換，不執行推斷
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))  # BGRA
    bgr = arr[:, :, :3].copy()                         # 去除 Alpha，轉為 BGR
    try:
        _frame_queue.put_nowait(bgr)
    except queue.Full:
        pass  # 主執行緒尚未完成上一幀推斷，丟棄此幀以保持即時性


def run_pipeline(
    bgr_frame: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    cfg: dict,
) -> tuple:
    """執行完整五階段推斷，回傳 (resized_image, left_fits, right_fits, widths, pitch_deg)。"""
    # BGR → RGB → PIL Image（PIDNet 模型期望 RGB 輸入）
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    resize_size = tuple(cfg["input"]["resize_size"])          # (512, 1024)
    min_slope   = cfg["lane_segmentation"]["min_slope"]
    min_len_near = cfg["lane_segmentation"]["min_segment_length_near"]
    min_len_far  = cfg["lane_segmentation"]["min_segment_length_far"]
    tolerance   = cfg["lane_segmentation"]["lane_band_tolerance"]
    extra_pts   = cfg["lane_fitting"]["extra_points_per_segment"]
    num_bands   = cfg["lane_fitting"]["num_bands"]
    num_samples = cfg["lane_fitting"]["num_samples"]
    f_x         = cfg["pitch_estimation"]["f_x"]
    f_y         = cfg["pitch_estimation"]["f_y"]   # 已覆蓋為 512
    w_real      = cfg["pitch_estimation"]["w_real"]

    # 階段 1：道路分割（PIDNet，argmax 取代 sigmoid+threshold）
    resized_image, pred_mask = predict_road_from_pil(
        model, pil_image, device, resize_size
    )
    masked_road, _ = apply_road_mask(resized_image, pred_mask)

    # 階段 2：車道線偵測（ELSED）
    segments = detect_lines_with_elsed(masked_road, min_len_near, min_len_far)

    # 階段 3：左右車道分類
    inner_left, inner_right = split_left_right_lines(
        segments,
        image_width=resized_image.width,
        min_slope=min_slope,
        img_height=resized_image.height,
        lane_band_tolerance=tolerance,
    )

    # 車道線不足時（路口、遮蔽等），回傳空結果
    if not inner_left or not inner_right:
        return resized_image, [], [], np.empty((0, 2)), None

    # 階段 4：車道擬合與車道寬度計算
    left_fits  = piecewise_linear_fit(
        collect_points_from_segments(inner_left,  extra_pts), num_bands
    )
    right_fits = piecewise_linear_fit(
        collect_points_from_segments(inner_right, extra_pts), num_bands
    )

    if not left_fits or not right_fits:
        return resized_image, [], [], np.empty((0, 2)), None

    widths = compute_lane_widths(left_fits, right_fits, num_samples)

    if len(widths) == 0:
        return resized_image, left_fits, right_fits, widths, None

    # 階段 5：俯仰角估計
    pitch_deg = estimate_pitch_from_widths(
        widths, f_x, f_y, resized_image.height, w_real
    )

    return resized_image, left_fits, right_fits, widths, pitch_deg


def render_display(
    resized_image,
    left_fits: list,
    right_fits: list,
    widths: np.ndarray,
    est_pitch,    # float 或 None
    gt_pitch: float,
    fps: float,
) -> np.ndarray:
    """繪製車道線擬合結果並疊加文字資訊，回傳 BGR display 影像。"""
    if left_fits and right_fits and len(widths) > 0:
        display = render_piecewise_fits_to_array(resized_image, left_fits, right_fits, widths)
    else:
        # 無車道線時直接轉換原始影像顯示
        display = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)

    h = display.shape[0]
    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness  = 2

    # 估計俯仰角（左上角，青色）
    est_text = f"Est. Pitch: {est_pitch:+.2f} deg" if est_pitch is not None else "Est. Pitch: --"
    cv2.putText(display, est_text, (10, 30), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

    # Ground Truth 俯仰角（左上角，深青色）
    cv2.putText(display, f"GT  Pitch: {gt_pitch:+.2f} deg",
                (10, 62), font, font_scale, (0, 200, 200), thickness, cv2.LINE_AA)

    # FPS（左下角，白色）
    cv2.putText(display, f"FPS: {fps:.1f}", (10, h - 12),
                font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return display


class _FPSCounter:
    """以滑動視窗平滑計算幀率。"""
    def __init__(self, window: int = 30):
        self._times: list[float] = []
        self._window = window

    def tick(self) -> None:
        self._times.append(time.perf_counter())
        if len(self._times) > self._window:
            self._times.pop(0)

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config_path)

    device = select_device(cfg["model"]["device"])
    print(f"[初始化] 使用裝置：{device}")

    model = load_pidnet(cfg["model"]["model_name"], cfg["model"]["weight_path"], device)
    print("[初始化] PIDNet 模型載入完成")

    client, world, vehicle, camera = setup_carla(args.host, args.port, args.timeout, args.map)
    print(f"[初始化] CARLA 連線成功，車輛已生成於 {vehicle.get_transform().location}")

    camera.listen(_camera_callback)
    print("[初始化] 相機開始串流，按 'q' 離開")

    fps_counter = _FPSCounter()
    cv2.namedWindow("CARLA Pitch Estimation", cv2.WINDOW_NORMAL)

    try:
        while True:
            # 等待下一幀（逾時 1 秒後繼續檢查）
            try:
                bgr_frame = _frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # 取得 Ground Truth 俯仰角
            gt_pitch: float = vehicle.get_transform().rotation.pitch

            try:
                resized_image, left_fits, right_fits, widths, est_pitch = run_pipeline(
                    bgr_frame, model, device, cfg
                )
            except Exception as exc:
                # 推斷例外時仍顯示原始幀，避免程式中斷
                print(f"[警告] 推斷發生例外：{exc}")
                display = bgr_frame.copy()
                cv2.putText(display, f"Error: {exc}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow("CARLA Pitch Estimation", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            fps_counter.tick()
            display = render_display(
                resized_image, left_fits, right_fits, widths,
                est_pitch, gt_pitch, fps_counter.fps
            )

            cv2.imshow("CARLA Pitch Estimation", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # 依序清理 CARLA Actor，確保不留下殘餘物件
        print("[清理] 停止相機串流...")
        camera.stop()
        camera.destroy()
        vehicle.set_autopilot(False)
        vehicle.destroy()
        cv2.destroyAllWindows()
        print("[清理] 已離開 CARLA 場景。")


if __name__ == "__main__":
    main()
