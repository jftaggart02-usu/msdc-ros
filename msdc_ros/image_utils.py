import cv2
import numpy as np
from sensor_msgs.msg import Image


def convert_rgb_msg_to_numpy(msg: Image, bgr: bool = True) -> np.ndarray:
    """Convert a ROS2 Image message to a numpy array."""

    # Check encoding type
    if msg.encoding not in ["bgr8", "rgb8"]:
        raise ValueError(f"Unsupported encoding type: {msg.encoding}")

    # 1. Create numpy array from data
    data = np.frombuffer(msg.data, dtype=np.uint8)
    
    # 2. Reshape to (height, step)
    data = data.reshape(msg.height, msg.step)
    
    # 3. Remove padding on the right (if any)
    # 4. Reshape to height x width x channels
    data = data[:, :msg.width*3].reshape(msg.height, msg.width, 3)

    # 5. Convert to BGR if needed for OpenCV
    if msg.encoding == "rgb8" and bgr:
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    if msg.encoding == "bgr8" and not bgr:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

    return data

def convert_depth_msg_to_numpy(msg: Image) -> np.ndarray:
    """Convert a ROS2 Image message containing depth data to a numpy array."""

    if msg.encoding != "16UC1":
        raise ValueError(f"Unsupported encoding type for depth image: {msg.encoding}")
    
    # 1. Create numpy array from data
    data = np.frombuffer(msg.data, dtype=np.uint16)
    
    # 2. Reshape to (height, step)
    data = data.reshape(msg.height, msg.step//2)
    
    # 3. Remove padding on the right (if any)
    data = data[:, :msg.width]  # 2 bytes per pixel for uint16

    assert data.shape == (msg.height, msg.width), f"Depth image data has incorrect shape after processing: {data.shape}, expected ({msg.height}, {msg.width})"

    # Swap bytes if the data is big-endian (ROS2 Image messages are typically little-endian, but we check just in case)
    if msg.is_bigendian:
        data = data.byteswap()

    return data