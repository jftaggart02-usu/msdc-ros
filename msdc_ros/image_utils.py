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

def convert_numpy_to_rgb_msg(image: np.ndarray, encoding: str = "rgb8") -> Image:
    """Convert a numpy array to a ROS2 Image message with RGB or BGR encoding.
    
    Args:
        image: A numpy array of shape (H, W, 3) containing the image data. Encoding must be in BGR format (as is standard for OpenCV).
        encoding: The desired encoding for the ROS Image message. Must be either "bgr8" or "rgb8".

    Returns:
        A ROS2 Image message containing the image data.
    """
    if encoding not in ["bgr8", "rgb8"]:
        raise ValueError(f"Unsupported encoding type: {encoding}")

    # Ensure the image has 3 channels
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must have 3 channels (H x W x 3)")

    # Convert to the correct color space if needed
    if encoding == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create Image message
    msg = Image()
    msg.height, msg.width, _ = image.shape
    msg.encoding = encoding
    msg.is_bigendian = False
    msg.step = msg.width * 3  # 3 bytes per pixel for RGB/BGR

    # Flatten the image data and convert to bytes
    msg.data = image.tobytes()

    return msg
