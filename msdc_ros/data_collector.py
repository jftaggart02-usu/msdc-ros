"""ROS2 node that subscribes to aligned RGB and Depth image topics, camera info messages, and steering control commands and saves them to a dataset for training a CNN."""

from pathlib import Path
import json
from itertools import combinations
import csv
import math

import numpy as np
import rclpy
from rclpy.node import Node, ParameterDescriptor
from rosmaster_r2_msgs.msg import AkmControl
from sensor_msgs.msg import Image, CameraInfo
import cv2
from msdc_ros.image_utils import convert_rgb_msg_to_numpy, convert_depth_msg_to_numpy


class DataCollector(Node):
    """A ROS2 node that subscribes to aligned RGB and Depth image topics, camera info messages, and steering control commands and saves them to a dataset for training a CNN.
    
    The created dataset is of the form:
    ```
    dataset_dir/
        rgb/
            00000000.png
            00000001.png
            ...
        depth/
            00000000.png
            00000001.png
            ...
        labels.csv
        intrinsics.json
    ```
    """

    def __init__(self):
        super().__init__('record_data')

        # Declare parameters
        self.declare_parameter(name="rgb_topic", value="/camera/camera/color/image_raw")
        self.declare_parameter(name="rgb_info_topic", value="/camera/camera/color/camera_info")
        self.declare_parameter(name="depth_topic", value="/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter(name="depth_info_topic", value="/camera/camera/aligned_depth_to_color/camera_info")
        self.declare_parameter(name="control_topic", value="/movement_control")
        self.declare_parameter(name="data_collect_rate", value=10.0, descriptor=ParameterDescriptor(description="Rate at which to collect data in Hz"))
        self.declare_parameter(name="dataset_dir", value="/tmp/dataset", descriptor=ParameterDescriptor(description="Directory to save collected dataset. Directory must not already exist."))
        self.declare_parameter(name="max_sync_err_ms", value=50.0, descriptor=ParameterDescriptor(description="The node will output a warning if the timestamps of the latest samples collected differ by more than this value."))
        self.declare_parameter(name="reject_bad_samples", value=True, descriptor=ParameterDescriptor(description="Whether to reject samples where the timestamps of the latest samples collected differ by more than max_sync_err_ms. If False, the node will still collect and save these samples but will output a warning."))

        # Set parameters
        self.rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        self.rgb_info_topic = self.get_parameter("rgb_info_topic").get_parameter_value().string_value
        self.depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        self.depth_info_topic = self.get_parameter("depth_info_topic").get_parameter_value().string_value
        self.control_topic = self.get_parameter("control_topic").get_parameter_value().string_value
        self.data_collect_period = 1.0 / self.get_parameter("data_collect_rate").get_parameter_value().double_value
        self.dataset_dir = Path(self.get_parameter("dataset_dir").get_parameter_value().string_value)
        self.max_sync_err_ms = self.get_parameter("max_sync_err_ms").get_parameter_value().double_value
        self.reject_bad_samples = self.get_parameter("reject_bad_samples").get_parameter_value().bool_value

        # Create dataset directory structure
        self.get_logger().info(f"Creating dataset directory at: {self.dataset_dir}")
        self.dataset_dir.mkdir(parents=True, exist_ok=False)
        self.dataset_dir.joinpath("rgb").mkdir()
        self.dataset_dir.joinpath("depth").mkdir()

        # Initialize CSV writer
        self.get_logger().info(f"Creating labels.csv file at: {self.dataset_dir / 'labels.csv'}")
        self.labels_file = open(self.dataset_dir / "labels.csv", "w", newline="")
        self.writer = csv.writer(self.labels_file)
        self.writer.writerow(["timestamp", "index", "steering_angle_rad"])

        # Create subscriptions
        self.sub_rgb = self.create_subscription(msg_type=Image, topic=self.rgb_topic, callback=self.rgb_cb, qos_profile=10)
        self.sub_depth = self.create_subscription(msg_type=Image, topic=self.depth_topic, callback=self.depth_cb, qos_profile=10)
        self.sub_control = self.create_subscription(msg_type=AkmControl, topic=self.control_topic, callback=self.control_cb, qos_profile=10)
        self.sub_rgb_info = self.create_subscription(msg_type=CameraInfo, topic=self.rgb_info_topic, callback=self.rgb_info_cb, qos_profile=10)
        self.sub_depth_info = self.create_subscription(msg_type=CameraInfo, topic=self.depth_info_topic, callback=self.depth_info_cb, qos_profile=10)

        # Create a timer for data collection
        self.data_collect_timer = self.create_timer(timer_period_sec=self.data_collect_period, callback=self.collect_data)

        # Store latest RGB, Depth, and Control messages
        self.latest_rgb: Image | None = None
        self.latest_rgb_timestamp: float | None = None
        self.latest_depth: Image | None = None
        self.latest_depth_timestamp: float | None = None
        self.latest_control: AkmControl | None = None
        self.latest_control_timestamp: float | None = None

        # Store first RGB and Depth info messages
        self.rgb_info: CameraInfo | None = None
        self.depth_info: CameraInfo | None = None

        # Keep track of whether we have written to intrinsics.json yet
        self.intrinsics_stored = False

        # Keep track of sample index
        self.sample_index = 0

        # Keep track of node start time so we can normalize timestamps in the dataset
        self.start_time_sec = self.get_clock().now().nanoseconds / 1e9

    def rgb_cb(self, msg: Image) -> None:
        """Save latest RGB message."""
        self.latest_rgb = msg
        self.latest_rgb_timestamp = self.get_current_time_sec()

    def depth_cb(self, msg: Image) -> None:
        """Save latest depth message."""
        self.latest_depth = msg
        self.latest_depth_timestamp = self.get_current_time_sec()

    def control_cb(self, msg: AkmControl) -> None:
        """Save latest control message."""
        self.latest_control = msg
        self.latest_control_timestamp = self.get_current_time_sec()

    def rgb_info_cb(self, msg: CameraInfo) -> None:
        """Save the first RGB Info message."""
        if not self.rgb_info:
            self.rgb_info = msg

    def depth_info_cb(self, msg: CameraInfo) -> None:
        """Save the first depth info message."""
        if not self.depth_info:
            self.depth_info = msg

    def collect_data(self) -> None:
        """Collect and save synchronized RGB, depth, and control data samples."""
        
        # If we have received depth and rgb info and also haven't yet written to intrinsics.json
        if not self.intrinsics_stored and self.rgb_info and self.depth_info:
            self.write_intrinsics()
            self.intrinsics_stored = True
            self.get_logger().info("Camera intrinsics have been written to intrinsics.json. Beginning data collection...")

        # Once intrinsics.json has been written, begin sample collection
        if self.intrinsics_stored and self.latest_rgb and self.latest_depth and self.latest_control:

            # Warn if timestamps of latest samples differ by more than max_sync_err_ms
            sync_err_ms = max([abs(t1 - t2) for t1, t2 in combinations([self.latest_control_timestamp, self.latest_depth_timestamp, self.latest_rgb_timestamp], 2)]) * 1000.0
            if sync_err_ms > self.max_sync_err_ms:
                if self.reject_bad_samples:
                    self.get_logger().warning(f"Rejecting sample due to sync error of {sync_err_ms} ms exceeding maximum sync error of {self.max_sync_err_ms} ms.")
                    return
                self.get_logger().warning(f"Sync error of {sync_err_ms} ms exceeds maximum sync error of {self.max_sync_err_ms} ms.")

            # Write depth and RGB images to files
            index_str = f"{self.sample_index:08d}"
            filename = index_str + ".png"
            cv2.imwrite(str(self.dataset_dir / "rgb" / filename), self.convert_rgb_msg_to_numpy(self.latest_rgb))
            cv2.imwrite(str(self.dataset_dir / "depth" / filename), self.convert_depth_msg_to_numpy(self.latest_depth))

            # Write steering angle (radians) to CSV
            self.writer.writerow([f"{self.latest_rgb_timestamp:0.9f}", index_str, f"{math.radians(self.latest_control.steering_angle):0.9f}"])

            self.sample_index += 1

    @staticmethod
    def serializeCameraInfo(camera_info: CameraInfo) -> dict:
        """Convert a ROS2 CameraInfo message to a serializable dictionary."""

        return {
            "frame_id": camera_info.header.frame_id,
            "width": camera_info.width,
            "height": camera_info.height,
            "distortion_model": camera_info.distortion_model,
            "d": list(camera_info.d),
            "k": list(camera_info.k),
            "r": list(camera_info.r),
            "p": list(camera_info.p),
            "binning_x": camera_info.binning_x,
            "binning_y": camera_info.binning_y,
            "roi": {
                "x_offset": camera_info.roi.x_offset,
                "y_offset": camera_info.roi.y_offset,
                "height": camera_info.roi.height,
                "width": camera_info.roi.width,
                "do_rectify": camera_info.roi.do_rectify
            }
        }

    def write_intrinsics(self) -> None:
        """Write RGB and depth camera intrinsics to a JSON file in the dataset directory."""

        json_path = self.dataset_dir / "intrinsics.json"
        intrinsics_data = {
            "rgb": self.serializeCameraInfo(self.rgb_info),
            "depth": self.serializeCameraInfo(self.depth_info)
        }
        with open(json_path, "w") as f:
            json.dump(intrinsics_data, f, indent=4)

    def get_current_time_sec(self) -> float:
        """Get the current ROS time in seconds."""

        return self.get_clock().now().nanoseconds / 1e9 - self.start_time_sec
    
    def cleanup(self) -> None:
        """Clean up any resources (e.g. open files) before shutting down the node."""

        if getattr(self, "labels_file", None) and not self.labels_file.closed:
            self.labels_file.close()

    def destroy_node(self) -> None:
        """Override the default destroy_node method to include cleanup logic."""

        self.cleanup()
        super().destroy_node()


def main():
    rclpy.init()
    node = DataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
