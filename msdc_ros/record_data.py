import csv
import json
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rosmaster_r2_msgs.msg import AkmControl
from sensor_msgs.msg import CameraInfo, Image


class RecordDataNode(Node):
	"""Record synchronized RGB/depth frames with steering labels."""

	def __init__(self) -> None:
		super().__init__("record_data")

		self.declare_parameter("record_fps", 10.0)
		self.declare_parameter("dataset_dir", "./dataset")
		self.declare_parameter("sync_max_skew_ms", 50.0)
		self.declare_parameter("rgb_topic", "/camera/camera/color/image_raw")
		self.declare_parameter("rgb_camera_info_topic", "/camera/camera/color/camera_info")
		self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
		self.declare_parameter("depth_camera_info_topic", "/camera/camera/aligned_depth_to_color/camera_info")
		self.declare_parameter("control_topic", "/movement_control")

		self.record_fps = float(self.get_parameter("record_fps").value)
		self.sync_max_skew_ms = float(self.get_parameter("sync_max_skew_ms").value)
		self.dataset_dir = Path(str(self.get_parameter("dataset_dir").value)).expanduser().resolve()

		if self.record_fps <= 0.0:
			raise ValueError("record_fps must be > 0")
		if self.sync_max_skew_ms < 0.0:
			raise ValueError("sync_max_skew_ms must be >= 0")

		self.sync_max_skew_ns = int(self.sync_max_skew_ms * 1e6)

		self.rgb_topic = str(self.get_parameter("rgb_topic").value)
		self.rgb_camera_info_topic = str(self.get_parameter("rgb_camera_info_topic").value)
		self.depth_topic = str(self.get_parameter("depth_topic").value)
		self.depth_camera_info_topic = str(self.get_parameter("depth_camera_info_topic").value)
		self.control_topic = str(self.get_parameter("control_topic").value)

		self._validate_output_directory()
		self.rgb_dir = self.dataset_dir / "rgb"
		self.depth_dir = self.dataset_dir / "depth"
		self.rgb_dir.mkdir(parents=True, exist_ok=True)
		self.depth_dir.mkdir(parents=True, exist_ok=True)

		self.labels_path = self.dataset_dir / "labels.csv"
		self.intrinsics_path = self.dataset_dir / "intrinsics.json"
		self.labels_file = open(self.labels_path, "w", newline="")
		self.labels_writer = csv.writer(self.labels_file)
		self.labels_writer.writerow(["timestamp", "index", "steering_angle_rad"])

		self.latest_rgb: Optional[Image] = None
		self.latest_depth: Optional[Image] = None
		self.latest_control: Optional[AkmControl] = None
		self.latest_rgb_recv_ns: Optional[int] = None
		self.latest_depth_recv_ns: Optional[int] = None
		self.latest_control_recv_ns: Optional[int] = None

		self.rgb_camera_info: Optional[CameraInfo] = None
		self.depth_camera_info: Optional[CameraInfo] = None
		self.intrinsics_written = False

		self.sample_index = 0
		self.sample_timer = None

		self.create_subscription(Image, self.rgb_topic, self._rgb_callback, 10)
		self.create_subscription(Image, self.depth_topic, self._depth_callback, 10)
		self.create_subscription(AkmControl, self.control_topic, self._control_callback, 10)
		self.create_subscription(CameraInfo, self.rgb_camera_info_topic, self._rgb_camera_info_callback, 10)
		self.create_subscription(CameraInfo, self.depth_camera_info_topic, self._depth_camera_info_callback, 10)

		self.get_logger().info("RecordData node initialized; waiting for one RGB and Depth camera_info message")

	def _validate_output_directory(self) -> None:
		if self.dataset_dir.exists() and any(self.dataset_dir.iterdir()):
			raise RuntimeError(f"dataset_dir '{self.dataset_dir}' already exists and is not empty (fail-fast)")
		self.dataset_dir.mkdir(parents=True, exist_ok=True)

	def _rgb_callback(self, msg: Image) -> None:
		self.latest_rgb = msg
		self.latest_rgb_recv_ns = self.get_clock().now().nanoseconds

	def _depth_callback(self, msg: Image) -> None:
		self.latest_depth = msg
		self.latest_depth_recv_ns = self.get_clock().now().nanoseconds

	def _control_callback(self, msg: AkmControl) -> None:
		self.latest_control = msg
		self.latest_control_recv_ns = self.get_clock().now().nanoseconds

	def _rgb_camera_info_callback(self, msg: CameraInfo) -> None:
		if self.rgb_camera_info is None:
			self.rgb_camera_info = msg
			self.get_logger().info("Received RGB camera_info")
			self._maybe_start_recording()

	def _depth_camera_info_callback(self, msg: CameraInfo) -> None:
		if self.depth_camera_info is None:
			self.depth_camera_info = msg
			self.get_logger().info("Received Depth camera_info")
			self._maybe_start_recording()

	def _maybe_start_recording(self) -> None:
		if self.intrinsics_written:
			return
		if self.rgb_camera_info is None or self.depth_camera_info is None:
			return

		self._write_intrinsics_json()
		timer_period = 1.0 / self.record_fps
		self.sample_timer = self.create_timer(timer_period, self._sampling_callback)
		self.get_logger().info(
			f"intrinsics.json written to {self.intrinsics_path}. Sampling started at {self.record_fps:.2f} Hz"
		)

	def _write_intrinsics_json(self) -> None:
		intrinsics_payload = {
			"rgb": self._camera_info_to_dict(self.rgb_camera_info),
			"depth": self._camera_info_to_dict(self.depth_camera_info),
		}
		with open(self.intrinsics_path, "w", encoding="utf-8") as intrinsics_file:
			json.dump(intrinsics_payload, intrinsics_file, indent=2)
		self.intrinsics_written = True

	@staticmethod
	def _camera_info_to_dict(camera_info: Optional[CameraInfo]) -> dict:
		if camera_info is None:
			return {}
		return {
			"topic_stamp": f"{camera_info.header.stamp.sec}.{camera_info.header.stamp.nanosec:09d}",
			"frame_id": camera_info.header.frame_id,
			"width": int(camera_info.width),
			"height": int(camera_info.height),
			"distortion_model": camera_info.distortion_model,
			"d": [float(v) for v in camera_info.d],
			"k": [float(v) for v in camera_info.k],
			"r": [float(v) for v in camera_info.r],
			"p": [float(v) for v in camera_info.p],
		}

	@staticmethod
	def _stamp_to_ns(image_msg: Image) -> int:
		return image_msg.header.stamp.sec * 1_000_000_000 + image_msg.header.stamp.nanosec

	@staticmethod
	def _row_aligned_uint8(msg: Image) -> np.ndarray:
		if msg.step <= 0:
			raise ValueError("Image step must be > 0")
		expected_bytes = msg.height * msg.step
		buffer = np.frombuffer(msg.data, dtype=np.uint8)
		if buffer.size < expected_bytes:
			raise ValueError(
				f"Image data buffer too small (got {buffer.size} bytes, expected {expected_bytes})"
			)
		return buffer[:expected_bytes].reshape((msg.height, msg.step))

	def _decode_color_to_bgr(self, msg: Image) -> np.ndarray:
		encoding = msg.encoding.lower()
		rows = self._row_aligned_uint8(msg)

		if encoding in {"bgr8", "rgb8"}:
			if msg.step < msg.width * 3:
				raise ValueError(f"Invalid step {msg.step} for {encoding} with width {msg.width}")
			image = rows[:, : msg.width * 3].reshape((msg.height, msg.width, 3))
			if encoding == "rgb8":
				image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			return image

		if encoding in {"bgra8", "rgba8"}:
			if msg.step < msg.width * 4:
				raise ValueError(f"Invalid step {msg.step} for {encoding} with width {msg.width}")
			image = rows[:, : msg.width * 4].reshape((msg.height, msg.width, 4))
			if encoding == "bgra8":
				return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
			return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

		if encoding == "mono8":
			if msg.step < msg.width:
				raise ValueError(f"Invalid step {msg.step} for mono8 with width {msg.width}")
			gray = rows[:, : msg.width].reshape((msg.height, msg.width))
			return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

		raise ValueError(f"Unsupported RGB image encoding: {msg.encoding}")

	def _decode_depth_uint16(self, msg: Image) -> np.ndarray:
		encoding = msg.encoding.lower()
		if encoding not in {"16uc1", "mono16"}:
			raise ValueError(f"Unsupported depth encoding: {msg.encoding}")

		if msg.step < msg.width * 2:
			raise ValueError(f"Invalid step {msg.step} for {encoding} with width {msg.width}")

		rows = self._row_aligned_uint8(msg)
		depth_bytes = rows[:, : msg.width * 2]
		depth_dtype = np.dtype(">u2") if bool(msg.is_bigendian) else np.dtype("<u2")
		depth = depth_bytes.reshape((msg.height, msg.width, 2)).view(depth_dtype).reshape((msg.height, msg.width))
		if depth.dtype.byteorder == ">":
			depth = depth.byteswap().newbyteorder()
		return depth.astype(np.uint16, copy=False)

	def _sampling_callback(self) -> None:
		if self.latest_rgb is None or self.latest_depth is None or self.latest_control is None:
			return

		if (
			self.latest_rgb_recv_ns is None
			or self.latest_depth_recv_ns is None
			or self.latest_control_recv_ns is None
		):
			return

		recv_times_ns = [self.latest_rgb_recv_ns, self.latest_depth_recv_ns, self.latest_control_recv_ns]
		skew_ns = max(recv_times_ns) - min(recv_times_ns)
		if skew_ns > self.sync_max_skew_ns:
			self.get_logger().warn(
				"Skipping sample: RGB/Depth/Control max skew exceeded "
				f"({skew_ns / 1e6:.2f} ms > {self.sync_max_skew_ms:.2f} ms)"
			)
			return

		try:
			rgb_bgr = self._decode_color_to_bgr(self.latest_rgb)
			depth_image = self._decode_depth_uint16(self.latest_depth)
		except Exception as exc:
			self.get_logger().warn(f"Skipping sample due to image conversion error: {exc}")
			return

		if depth_image.dtype != "uint16":
			self.get_logger().warn(
				f"Skipping sample: depth dtype must be uint16 for raw depth PNG, got {depth_image.dtype}"
			)
			return

		self.sample_index += 1
		sample_name = f"{self.sample_index:08d}.png"

		rgb_path = self.rgb_dir / sample_name
		depth_path = self.depth_dir / sample_name

		rgb_ok = cv2.imwrite(str(rgb_path), rgb_bgr)
		depth_ok = cv2.imwrite(str(depth_path), depth_image)

		if not rgb_ok or not depth_ok:
			self.get_logger().warn(f"Skipping label write: failed to write images for sample {self.sample_index:08d}")
			self.sample_index -= 1
			return

		timestamp = f"{self.latest_rgb_recv_ns}"  # Use RGB receive time as the sample timestamp
		steering_angle_rad = math.radians(float(self.latest_control.steering_angle))
		self.labels_writer.writerow([timestamp, f"{self.sample_index:08d}", f"{steering_angle_rad:.9f}"])
		self.labels_file.flush()

	def destroy_node(self) -> bool:
		try:
			if hasattr(self, "labels_file") and not self.labels_file.closed:
				self.labels_file.close()
		finally:
			return super().destroy_node()


def main(args=None) -> None:
	rclpy.init(args=args)
	node = None
	try:
		node = RecordDataNode()
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		if node is not None:
			node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main()
