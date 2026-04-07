from rclpy.node import Node
from sensor_msgs.msg import Joy
from rosmaster_r2_akm_driver.msg import AkmControl


class TeleopNode(Node):
    """ROS2 node that subscribes to joystick messages and publishes control commands based on the joystick input."""

    def __init__(self) -> None:
        super().__init__("teleop_node")

        # Declare parameters
        self.declare_parameter("joy_topic", value="/joy")
        self.declare_parameter("control_topic", value="/movement_control")
        self.declare_parameter("velocity_axis", value=1)
        self.declare_parameter("steering_axis", value=2)
        self.declare_parameter("max_velocity", value=0.3)
        self.declare_parameter("max_steering_angle", value=45)
        self.declare_parameter("steering_sensitivity", value=0.5)

        # Set parameters
        self.joy_topic = self.get_parameter("joy_topic").get_parameter_value().string_value
        self.control_topic = self.get_parameter("control_topic").get_parameter_value().string_value
        self.velocity_axis = self.get_parameter("velocity_axis").get_parameter_value().integer_value
        self.steering_axis = self.get_parameter("steering_axis").get_parameter_value().integer_value
        self.max_velocity = self.get_parameter("max_velocity").get_parameter_value().double_value
        self.max_steering_angle = self.get_parameter("max_steering_angle").get_parameter_value().double_value
        self.steering_sensitivity = self.get_parameter("steering_sensitivity").get_parameter_value().double_value

        # Create publishers and subscribers
        self.sub_joy = self.create_subscription(msg_type=Joy, topic=self.joy_topic, callback=self.joy_callback, qos_profile=10)
        self.pub_control = self.create_publisher(msg_type=AkmControl, topic=self.control_topic, qos_profile=10)
        
    def joy_callback(self, msg: Joy) -> None:
        """Each time a joystick value is received, convert to control commands and publish the result.
        
        Args:
            msg: the message received
        """

        # Extract steering and throttle axis values from joy message. Axis values should be -1.0 to 1.0.
        velocity_cmd = msg.axes[self.velocity_axis]
        steering_cmd = msg.axes[self.steering_axis]
        if abs(velocity_cmd) > 1.0 or abs(steering_cmd) > 1.0:
            raise ValueError("Joystick axis value is outside the expected range of -1.0 to 1.0!")

        # Apply joystick curve to steering.
        steering_cmd = self.apply_joystick_curve(steering_cmd, self.steering_sensitivity)

        # Map normalized axis values to actual commands.
        steering_cmd = self.map_value(steering_cmd, -1.0, 1.0, -self.max_steering_angle, self.max_steering_angle)
        velocity_cmd = self.map_value(velocity_cmd, -1.0, 1.0, -self.max_velocity, self.max_velocity)

        # Publish to control topic.
        control_msg = AkmControl(steering_angle=int(steering_cmd), velocity=velocity_cmd)
        self.pub_control.publish(control_msg)

    @staticmethod
    def apply_joystick_curve(value: float, alpha: float) -> float:
        """Apply hybrid cubic/linear sensitivity curve to input value.
        
        Args:
            value: The value to apply the curve to. Must be in the range [-1.0, 1.0].
            alpha: A parameter that determines the shape of the sensitivity curve. Must be in the range [0.0, 1.0]. 0.0 results in a pure linear curve, 1.0 results in a pure cubic curve, and values in between create a blend of the two.
       
        Returns:
            The value after applying the sensitivity curve, which will also be in the range [-1.0, 1.0].
        """

        if abs(value) > 1.0:
            raise ValueError("Value should be in the range [-1.0, 1.0]")
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("Alpha should be in the range [0.0, 1.0]")
        
        return (1-alpha) * value + (alpha) * value ** 3

    @staticmethod
    def map_value(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
        """Map a value from one range to another.
        
        Args:
            value: The value to map. Must be in the range [in_min, in_max].
            in_min: The minimum value of the input range.
            in_max: The maximum value of the input range.
            out_min: The minimum value of the output range.
            out_max: The maximum value of the output range.

        Returns:
            The value mapped to the output range.
        """
        if (value > in_max) or (value < in_min):
            raise ValueError("Value provided is outside the input range!")
        
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    
