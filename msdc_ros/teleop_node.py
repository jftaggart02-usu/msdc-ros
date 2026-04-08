"""ROS2 node that subscribes to joystick messages and publishes control commands based on the joystick input."""

import rclpy
from rclpy.node import Node, ParameterDescriptor
from sensor_msgs.msg import Joy
from rosmaster_r2_akm_driver.msg import AkmControl


class TeleopNode(Node):
    """ROS2 node that subscribes to joystick messages and publishes control commands based on the joystick input."""

    def __init__(self) -> None:
        super().__init__("teleop_node")

        # Declare parameters
        self.declare_parameter("joy_topic", value="/joy")
        self.declare_parameter("control_topic", value="/movement_control")
        self.declare_parameter("velocity_axis", value=1, descriptor=ParameterDescriptor(description="The index of the joystick axis to use for velocity control. Should be an integer corresponding to the desired axis in the Joy message."))
        self.declare_parameter("steering_axis", value=2, descriptor=ParameterDescriptor(description="The index of the joystick axis to use for steering control. Should be an integer corresponding to the desired axis in the Joy message."))
        self.declare_parameter("max_velocity", value=0.3, descriptor=ParameterDescriptor(description="The magnitude of the maximum velocity (m/s) that can be commanded. Should be a float in the range [0, 1.8]."))
        self.declare_parameter("max_steering_angle", value=45, descriptor=ParameterDescriptor(description="The magnitude of the maximum steering angle (degrees) that can be commanded. Should be a float in the range [0, 45] degrees."))
        self.declare_parameter("steering_sensitivity", value=0.5, descriptor=ParameterDescriptor(description="The sensitivity of the steering control. Should be a float between 0 and 1."))

        # Set parameters
        self.joy_topic = self.get_parameter("joy_topic").get_parameter_value().string_value
        self.control_topic = self.get_parameter("control_topic").get_parameter_value().string_value
        self.velocity_axis = self.get_parameter("velocity_axis").get_parameter_value().integer_value
        self.steering_axis = self.get_parameter("steering_axis").get_parameter_value().integer_value
        self.max_velocity = self.get_parameter("max_velocity").get_parameter_value().double_value
        self.max_steering_angle = self.get_parameter("max_steering_angle").get_parameter_value().double_value
        self.steering_sensitivity = self.get_parameter("steering_sensitivity").get_parameter_value().double_value
        
        # Validate parameters and log warnings if they are outside expected ranges
        if self.velocity_axis < 0:
            self.get_logger().warning(f"Velocity axis index of {self.velocity_axis} is negative! This may cause an IndexError when processing joystick messages.")
        if self.steering_axis < 0:
            self.get_logger().warning(f"Steering axis index of {self.steering_axis} is negative! This may cause an IndexError when processing joystick messages.")
        if not 0 <= self.max_velocity <= 1.8:
            self.get_logger().warning(f"Max velocity of {self.max_velocity} is outside the acceptable range! Clamping to [0, 1.8] m/s.")
            self.max_velocity = self.clamp(self.max_velocity, 0, 1.8)
        if not 0 <= self.max_steering_angle <= 45:
            self.get_logger().warning(f"Max steering angle of {self.max_steering_angle} is outside the acceptable range! Clamping to [0, 45] degrees.")
            self.max_steering_angle = self.clamp(self.max_steering_angle, 0, 45)
        if not 0 <= self.steering_sensitivity <= 1:
            self.get_logger().warning(f"Steering sensitivity of {self.steering_sensitivity} is outside the acceptable range! Clamping to [0, 1].")
            self.steering_sensitivity = self.clamp(self.steering_sensitivity, 0, 1)

        # Create publishers and subscribers
        self.sub_joy = self.create_subscription(msg_type=Joy, topic=self.joy_topic, callback=self.joy_callback, qos_profile=10)
        self.pub_control = self.create_publisher(msg_type=AkmControl, topic=self.control_topic, qos_profile=10)
        
    def joy_callback(self, msg: Joy) -> None:
        """Each time a joystick value is received, convert to control commands and publish the result.
        
        Args:
            msg: the message received
        """

        # Extract steering and throttle axis values from joy message. Axis values should be -1.0 to 1.0.
        try:
            velocity_cmd = msg.axes[self.velocity_axis]
            steering_cmd = msg.axes[self.steering_axis]
        except IndexError as e:
            self.get_logger().error(f"Joystick message does not contain expected axes. Error: {e}")
            return

        # Apply joystick curve to steering.
        steering_cmd = self.apply_joystick_curve(steering_cmd, self.steering_sensitivity)

        # Map normalized axis values to actual commands.
        steering_cmd = self.map_value(steering_cmd, -1.0, 1.0, -self.max_steering_angle, self.max_steering_angle)
        velocity_cmd = self.map_value(velocity_cmd, -1.0, 1.0, -self.max_velocity, self.max_velocity)

        # Publish to control topic.
        control_msg = AkmControl(steering_angle=int(round(steering_cmd)), velocity=velocity_cmd)
        self.pub_control.publish(control_msg)

    def map_value(self, value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
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
        new_value = (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        if not out_min <= new_value <= out_max:
            self.get_logger().warning(f"Mapped value of {new_value} is outside the expected range. Clamping to [{out_min}, {out_max}]!")
            new_value = self.clamp(new_value, out_min, out_max)
        return new_value

    @staticmethod
    def apply_joystick_curve(value: float, alpha: float) -> float:
        """Apply hybrid cubic/linear sensitivity curve to input value.
        
        Args:
            value: The value to apply the curve to. Should be in the range [-1.0, 1.0].
            alpha: A parameter that determines the shape of the sensitivity curve. Should be in the range [0.0, 1.0]. 0.0 results in a pure linear curve, 1.0 results in a pure cubic curve, and values in between create a blend of the two.
       
        Returns:
            The value after applying the sensitivity curve, which will also be in the range [-1.0, 1.0].
        """
        return (1-alpha) * value + (alpha) * value ** 3
            
    @staticmethod
    def clamp(value: float, min_value: float, max_value: float) -> float:
        """Clamp a value to a specified range.
        
        Args:
            value: The value to clamp.
            min_value: The minimum allowed value.
            max_value: The maximum allowed value.

        Returns:
            The clamped value, which will be in the range [min_value, max_value].
        """
        return max(min(value, max_value), min_value)
    
    
if __name__ == "__main__":
    rclpy.init()
    node = TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
