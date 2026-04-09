"""A ROS2 Node that subscribes to the /image_raw topic,
runs a forward pass through the SteeringNet,
and publishes the predicted steering angle and a constant velocity command to the /movement_control topic
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import ImageMsg
import torch
from PIL import Image
from torchvision import transforms
from msdc_core.steering_net.steering_net import SteeringNet
from rosmaster_r2_msgs.msg import AkmControl
from msdc_ros.image_utils import convert_rgb_msg_to_numpy


class SteeringController(Node):
    def __init__(self):
        super().__init__('steering_controller')

        # Declare parameters
        self.declare_parameter(name='model_path', value='path/to/steering_net_model.pth')
        self.declare_parameter(name='velocity', value=0.3)
        self.declare_parameter(name='image_topic', value='/camera/camera/color/image_raw')
        self.declare_parameter(name='control_topic', value='/movement_control')

        # Set parameters
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.velocity = self.get_parameter('velocity').get_parameter_value().double_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.control_topic = self.get_parameter('control_topic').get_parameter_value().string_value

        # Load the SteeringNet model
        self.model = SteeringNet()
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        # Define an image transform to normalize the image and convert it to a tensor
        self.transform = transforms.Compose([transforms.Resize((240, 320)), transforms.ToTensor()])

        # Subscribe to the /image_raw topic
        self.image_sub = self.create_subscription(
            msg_type=ImageMsg,
            topic=self.image_topic,
            callback=self.publish_control_command,
            qos_profile=1
        )

        # Publish to the /movement_control topic
        self.control_pub = self.create_publisher(
            msg_type=AkmControl,
            topic=self.control_topic,
            qos_profile=1
        )

    def image_to_tensor(self, image_msg: ImageMsg) -> torch.Tensor:
        """Converts a ROS Image message to a normalized PyTorch tensor suitable for input to the SteeringNet model."""
        # Convert the ROS Image message to a numpy array
        image_np = convert_rgb_msg_to_numpy(image_msg)

        # Convert the numpy array to a PIL Image
        image_pil = Image.fromarray(image_np).convert('RGB')

        # Apply the defined transforms to get a tensor of shape (C, H, W) with values in [0, 1]
        image_tensor = self.transform(image_pil)

        return image_tensor

    def publish_control_command(self, msg: ImageMsg):
        # Convert the incoming ROS Image message to a tensor
        image_tensor = self.image_to_tensor(msg)

        # Add a batch dimension to the tensor (shape becomes (1, C, H, W))
        image_tensor = image_tensor.unsqueeze(0)

        # Run a forward pass through the model to get the predicted steering angle
        with torch.no_grad():
            predicted_steering_angle = self.model(image_tensor).item()

        # Create an AkmControl message with the predicted steering angle and constant velocity
        control_msg = AkmControl()
        control_msg.steering_angle = int(round(math.degrees(predicted_steering_angle)))
        control_msg.velocity = self.velocity

        # Publish the control command
        self.control_pub.publish(control_msg)
        
        