"""A ROS2 Node that reads from a .mp4 video file and publishes the frames as sensor_msgs/Image messages."""

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from msdc_ros.image_utils import convert_numpy_to_rgb_msg

class VideoPublisher(Node):
    def __init__(self):
        super().__init__("video_publisher")

        # Declare parameters
        self.declare_parameter(name='video_path', value="path/to/your/video.mp4")
        self.declare_parameter(name='topic_name', value="/camera/camera/color/image_raw")
        self.declare_parameter(name='default_publish_rate', value=10.0)  # Default rate (in Hz) to publish frames if video FPS cannot be determined

        # Set parameters
        self.video_path = self.get_parameter('video_path').get_parameter_value().string_value
        self.topic_name = self.get_parameter('topic_name').get_parameter_value().string_value
        self.default_publish_rate = self.get_parameter('default_publish_rate').get_parameter_value().double_value

        # Create a VideoCapture object to read from the video file
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            err_msg = f"Failed to open video file: {self.video_path}"
            self.get_logger().error(err_msg)
            raise RuntimeError(err_msg)
        
        # Get the FPS from the video to set the timer rate
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = self.default_publish_rate

        # Set up timer to publish frames at the video's FPS
        self.publisher = self.create_publisher(Image, self.topic_name, 10)
        self.timer = self.create_timer(1.0 / fps, self.publish_frame)

        self.get_logger().info(f"VideoPublisher initialized with video: {self.video_path}, publishing to topic: {self.topic_name} at {fps} Hz")

    def publish_frame(self):
        # Read a frame from the video
        ret, frame = self.cap.read()

        # Restart when end of video is reached
        if not ret:
            self.get_logger().info("End of video reached, restarting.")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the video
            return

        # Convert the frame to a ROS Image message and publish it
        msg = convert_numpy_to_rgb_msg(frame, encoding="rgb8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_frame"
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    video_publisher = VideoPublisher()

    try:
        rclpy.spin(video_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        video_publisher.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
