"""
carla_module/get_carlaDataset.py
CARLA 資料集蒐集腳本：對齊 RGB 影像與 IMU 資料

使用方式：
    uv run python carla_module/get_carlaDataset.py [--host HOST] [--port PORT]
        [--speed KMH] [--camera-fps N] [--z-offset N]
        [--align-frames N] [--warmup-frames N]

狀態機流程：
    生成車輛
       ↓
    [ALIGNING]   TM autopilot 對齊車道（連續 --align-frames 幀 steer < 0.05）
       ↓
    [WARMUP]     關閉 TM，P 控制直行，等 --warmup-frames 幀讓物理穩定
       ↓
    [COLLECTING] steer=0 P 控制，存圖 + IMU，按 'q' 停止

儲存路徑：根目錄 carla_dataset_{Map}_{YYYYMMDD_HHMMSS}/
    ├── images/  000000.png, 000001.png, ...
    └── measurements.csv
"""

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from utils.env_setup import setup_env
setup_env()

_whl = os.environ.get("CARLA_WHL_PATH", "")
if _whl:
    _carla_api_dir = str(pathlib.Path(_whl).parent.parent)
    if _carla_api_dir not in sys.path:
        sys.path.insert(0, _carla_api_dir)

import argparse
import csv
import datetime
import queue
from dataclasses import dataclass
from typing import Optional

import carla
import cv2
import numpy as np


# ── 相機規格（比照 get_data.py）────────────────────────────────────────────────
IMG_WIDTH     = 1280
IMG_HEIGHT    = 720
CAMERA_HEIGHT = 1.08
PHYSICS_WARMUP_TICKS = 30   # 生成後讓物理系統落穩的預熱 tick 數


# ── 狀態機 ────────────────────────────────────────────────────────────────────

class State:
    ALIGNING   = "ALIGNING"    # TM autopilot 對齊車道中
    WARMUP     = "WARMUP"      # 關閉 TM 後等物理穩定
    COLLECTING = "COLLECTING"  # 手動 P 控制，開始存資料


# ── 資料結構 ──────────────────────────────────────────────────────────────────

@dataclass
class ImuSample:
    timestamp: float
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    compass: float


# ── 引數 ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CARLA 資料集蒐集")
    p.add_argument("--host",          default="127.0.0.1", help="CARLA 伺服器位址")
    p.add_argument("--port",          type=int,   default=2000,  help="埠號")
    p.add_argument("--timeout",       type=float, default=20.0,  help="連線逾時秒數")
    p.add_argument("--speed",         type=float, default=30.0,  help="目標車速 km/h")
    p.add_argument("--camera-fps",    type=int,   default=20,    help="同步模式 FPS（預設：20）")
    p.add_argument("--z-offset",      type=float, default=0.0,
                   help="spectator z 偏移量（預設 0，依場景高度調整）")
    p.add_argument("--align-frames",  type=int,   default=20,
                   help="連續幾幀 steer<0.05 才視為對齊完成（預設：20）")
    p.add_argument("--warmup-frames", type=int,   default=10,
                   help="切換 TM→手動後再等幾幀才開始存檔（預設：10）")
    return p.parse_args()


# ── 資料集寫入器 ───────────────────────────────────────────────────────────────

class DatasetWriter:
    _CSV_HEADER = [
        "frame_id", "camera_ts",
        "accel_x",  "accel_y",  "accel_z",
        "gyro_x",   "gyro_y",   "gyro_z",
        "compass",  "imu_ts",
        "gt_pitch_deg", "gt_yaw_deg",  "gt_roll_deg",
        "gt_loc_x",     "gt_loc_y",    "gt_loc_z",
        "gt_speed_mps",
    ]

    def __init__(self, root: pathlib.Path, map_name: str) -> None:
        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = root / "output" / f"carla_dataset_{map_name}_{ts_str}"
        self.img_dir  = self.save_dir / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)

        csv_path = self.save_dir / "measurements.csv"
        self._csv_f  = open(csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv_f)
        self._writer.writerow(self._CSV_HEADER)
        self._count = 0
        print(f"[資料集] 儲存至：{self.save_dir}")

    @property
    def count(self) -> int:
        return self._count

    def save(
        self,
        image:     "carla.Image",
        imu:       Optional[ImuSample],
        transform: "carla.Transform",
        speed_mps: float,
    ) -> None:
        img_path = str(self.img_dir / f"{self._count:06d}.png")
        image.save_to_disk(img_path)

        rot = transform.rotation
        loc = transform.location

        if imu is not None:
            row = [
                self._count,          f"{image.timestamp:.6f}",
                f"{imu.accel_x:.6f}", f"{imu.accel_y:.6f}", f"{imu.accel_z:.6f}",
                f"{imu.gyro_x:.6f}",  f"{imu.gyro_y:.6f}",  f"{imu.gyro_z:.6f}",
                f"{imu.compass:.4f}", f"{imu.timestamp:.6f}",
                f"{rot.pitch:.4f}",   f"{rot.yaw:.4f}",      f"{rot.roll:.4f}",
                f"{loc.x:.4f}",       f"{loc.y:.4f}",        f"{loc.z:.4f}",
                f"{speed_mps:.4f}",
            ]
        else:
            row = [
                self._count, f"{image.timestamp:.6f}",
                "", "", "", "", "", "", "", "",
                f"{rot.pitch:.4f}", f"{rot.yaw:.4f}", f"{rot.roll:.4f}",
                f"{loc.x:.4f}",     f"{loc.y:.4f}",  f"{loc.z:.4f}",
                f"{speed_mps:.4f}",
            ]

        self._writer.writerow(row)
        self._csv_f.flush()
        self._count += 1

    def close(self) -> None:
        self._csv_f.close()
        print(f"[資料集] 共儲存 {self._count} 幀。路徑：{self.save_dir}")


# ── 車輛控制 ──────────────────────────────────────────────────────────────────

def apply_straight_control(vehicle: carla.Vehicle, target_mps: float) -> None:
    """P 控制器維持速度，steer=0 保持直行。"""
    vel   = vehicle.get_velocity()
    speed = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5
    error = target_mps - speed

    throttle = float(np.clip(0.3 + error * 0.15, 0.0, 1.0))
    brake    = float(np.clip(-error * 0.1,        0.0, 1.0))

    vehicle.apply_control(carla.VehicleControl(
        throttle=throttle,
        steer=0.0,
        brake=brake,
        hand_brake=False,
        manual_gear_shift=False,
    ))


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main() -> None:
    args     = parse_args()
    root_dir = pathlib.Path(__file__).parent.parent

    # ── 連線 ──────────────────────────────────────────────────────────────────
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world     = client.get_world()
    carla_map = world.get_map()
    print(f"[初始化] 連線至地圖：{carla_map.name}")
    bp_lib = world.get_blueprint_library()

    # ── 生成車輛（從 spectator 當前位置生成）─────────────────────────────────
    vehicle_bp      = bp_lib.find("vehicle.tesla.model3")
    spectator       = world.get_spectator()
    spawn_transform = spectator.get_transform()
    spawn_transform.location.z  += args.z_offset
    spawn_transform.rotation.pitch = 0.0
    spawn_transform.rotation.roll  = 0.0

    vehicle: carla.Vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)
    if vehicle is None:
        raise RuntimeError(
            "無法在 spectator 位置生成車輛，請移動鏡頭到可通行路面，"
            "或以 --z-offset 調整偏移量（預設 0）"
        )
    vehicle.set_autopilot(False)
    print(f"[初始化] 車輛生成於：{spawn_transform.location}")

    TARGET_SPEED_MPS = args.speed / 3.6

    # ── 同步模式 ──────────────────────────────────────────────────────────────
    settings = world.get_settings()
    settings.synchronous_mode   = True
    settings.fixed_delta_seconds = 1.0 / args.camera_fps
    world.apply_settings(settings)

    # TM 也要進同步模式，否則與 world.tick() 衝突
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    # ── 感測器 ────────────────────────────────────────────────────────────────
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(IMG_WIDTH))
    cam_bp.set_attribute("image_size_y", str(IMG_HEIGHT))
    cam_bp.set_attribute("fov", "90")
    cam_tf = carla.Transform(carla.Location(x=1.5, z=CAMERA_HEIGHT))
    camera: carla.Sensor = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)

    imu_bp = bp_lib.find("sensor.other.imu")
    imu_sensor: carla.Sensor = world.spawn_actor(
        imu_bp, carla.Transform(), attach_to=vehicle
    )

    image_queue: queue.Queue = queue.Queue()
    imu_queue:   queue.Queue = queue.Queue()
    camera.listen(image_queue.put)
    imu_sensor.listen(imu_queue.put)

    # ── 物理預熱 ──────────────────────────────────────────────────────────────
    print(f"[初始化] 物理預熱 {PHYSICS_WARMUP_TICKS} ticks...")
    for _ in range(PHYSICS_WARMUP_TICKS):
        world.tick()

    # ── 啟動 TM 對齊 ──────────────────────────────────────────────────────────
    vehicle.set_autopilot(True, tm.get_port())
    tm.ignore_lights_percentage(vehicle, 100.0)
    tm.ignore_signs_percentage(vehicle,  100.0)
    tm.set_desired_speed(vehicle, args.speed)

    state          = State.ALIGNING
    align_counter  = 0
    warmup_counter = 0

    map_name = carla_map.name.split("/")[-1]
    writer   = DatasetWriter(root_dir, map_name)
    win_name = "CARLA Dataset Collection"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    print(f"[{state}] TM 對齊車道中（需連續 {args.align_frames} 幀 steer < 0.05）")

    try:
        while True:
            target_frame = world.tick()

            # ── 取出與本 tick 對齊的相機幀 ────────────────────────────────────
            image: Optional[carla.Image] = None
            try:
                while True:
                    temp = image_queue.get(timeout=2.0)
                    if temp.frame == target_frame:
                        image = temp
                        break
                    if temp.frame > target_frame:
                        break
            except queue.Empty:
                print(f"[警告] 幀 {target_frame} 影像遺失，跳過")
                continue

            if image is None:
                continue

            # ── 取出本 tick IMU ────────────────────────────────────────────────
            imu_raw: Optional[carla.IMUMeasurement] = None
            while not imu_queue.empty():
                imu_raw = imu_queue.get_nowait()

            imu_sample: Optional[ImuSample] = None
            if imu_raw is not None:
                imu_sample = ImuSample(
                    timestamp=imu_raw.timestamp,
                    accel_x=imu_raw.accelerometer.x,
                    accel_y=imu_raw.accelerometer.y,
                    accel_z=imu_raw.accelerometer.z,
                    gyro_x=imu_raw.gyroscope.x,
                    gyro_y=imu_raw.gyroscope.y,
                    gyro_z=imu_raw.gyroscope.z,
                    compass=imu_raw.compass,
                )

            transform = vehicle.get_transform()
            vel       = vehicle.get_velocity()
            speed_mps = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5

            # ── 狀態機 ────────────────────────────────────────────────────────
            if state == State.ALIGNING:
                steer = vehicle.get_control().steer
                if abs(steer) < 0.05:
                    align_counter += 1
                else:
                    align_counter = 0   # 不連續就重置

                if align_counter >= args.align_frames:
                    vehicle.set_autopilot(False)
                    state = State.WARMUP
                    warmup_counter = 0
                    print(f"[{state}] TM 已關閉，等待 {args.warmup_frames} 幀物理穩定...")

            elif state == State.WARMUP:
                apply_straight_control(vehicle, TARGET_SPEED_MPS)
                warmup_counter += 1
                if warmup_counter >= args.warmup_frames:
                    state = State.COLLECTING
                    print(f"[{state}] 開始存檔！按 'q' 停止")

            elif state == State.COLLECTING:
                apply_straight_control(vehicle, TARGET_SPEED_MPS)
                writer.save(image, imu_sample, transform, speed_mps)

            # ── 顯示視窗 ──────────────────────────────────────────────────────
            arr     = np.frombuffer(image.raw_data, dtype=np.uint8)
            display = arr.reshape((image.height, image.width, 4))[:, :, :3].copy()
            h, font = display.shape[0], cv2.FONT_HERSHEY_SIMPLEX

            state_color = {
                State.ALIGNING:   (0, 165, 255),   # 橘
                State.WARMUP:     (0, 255, 255),    # 黃
                State.COLLECTING: (0, 255, 0),      # 綠
            }[state]

            cv2.putText(display,
                        f"[{state}]  Saved: {writer.count}",
                        (10, 30),     font, 0.8, state_color,      2, cv2.LINE_AA)
            cv2.putText(display,
                        f"GT Pitch: {transform.rotation.pitch:+.2f} deg",
                        (10, 62),     font, 0.8, (0, 255, 255),    2, cv2.LINE_AA)
            cv2.putText(display,
                        f"Speed: {speed_mps * 3.6:.1f} km/h  "
                        f"Steer: {vehicle.get_control().steer:+.3f}",
                        (10, 94),     font, 0.8, (255, 255, 0),    2, cv2.LINE_AA)
            cv2.putText(display,
                        "Press Q to stop",
                        (10, h - 12), font, 0.7, (200, 200, 200),  2, cv2.LINE_AA)
            cv2.imshow(win_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        print("[清理] 恢復非同步模式...")
        s = world.get_settings()
        s.synchronous_mode   = False
        s.fixed_delta_seconds = None
        world.apply_settings(s)
        tm.set_synchronous_mode(False)

        camera.stop()
        imu_sensor.stop()
        camera.destroy()
        imu_sensor.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()
        writer.close()


if __name__ == "__main__":
    main()
