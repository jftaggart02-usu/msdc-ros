import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from rosmaster_r2_msgs.msg import AkmControl


class TeleopNode(Node):
    """ROS2 node that maps joystick input to AkmControl commands."""

    def __init__(self):
        super().__init__("teleop_node")

        # Declare parameters with default values
        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("control_topic", "/rm1/movement_control")
        self.declare_parameter("velocity_axis", 1)
        self.declare_parameter("steering_axis", 2)
        self.declare_parameter("max_velocity", 0.5)
        self.declare_parameter("max_steering_angle", 45)
        self.declare_parameter("joystick_sensitivity_curve", "linear")
        self.declare_parameter("joystick_sensitivity_alpha", 0.6)

        # Get parameter values
        joy_topic = self.get_parameter("joy_topic").value
        control_topic = self.get_parameter("control_topic").value
        self.velocity_axis = self.get_parameter("velocity_axis").value
        self.steering_axis = self.get_parameter("steering_axis").value
        self.max_velocity = self.get_parameter("max_velocity").value
        self.max_steering_angle = self.get_parameter("max_steering_angle").value
        self.joystick_sensitivity_curve = self.get_parameter("joystick_sensitivity_curve").value.lower()
        self.joystick_sensitivity_alpha = float(self.get_parameter("joystick_sensitivity_alpha").value)

        if self.joystick_sensitivity_curve not in {"linear", "nonlinear"}:
            self.get_logger().warn(
                f"Invalid joystick_sensitivity_curve '{self.joystick_sensitivity_curve}'. "
                "Falling back to 'linear'."
            )
            self.joystick_sensitivity_curve = "linear"

        if not 0.0 <= self.joystick_sensitivity_alpha <= 1.0:
            clamped_alpha = min(max(self.joystick_sensitivity_alpha, 0.0), 1.0)
            self.get_logger().warn(
                f"joystick_sensitivity_alpha {self.joystick_sensitivity_alpha} out of range [0.0, 1.0]. "
                f"Clamping to {clamped_alpha}."
            )
            self.joystick_sensitivity_alpha = clamped_alpha

        # Create subscriber and publisher
        self.joy_sub = self.create_subscription(Joy, joy_topic, self.joy_callback, 10)

        self.control_pub = self.create_publisher(AkmControl, control_topic, 10)

        self.get_logger().info(f"Teleop node initialized: subscribing to {joy_topic}, " f"publishing to {control_topic}")

    def _apply_sensitivity_curve(self, input_value: float) -> float:
        """Apply joystick sensitivity shaping to a normalized joystick input."""
        if self.joystick_sensitivity_curve == "linear":
            return input_value

        alpha = self.joystick_sensitivity_alpha
        return ((1.0 - alpha) * input_value) + (alpha * (input_value**3))

    def joy_callback(self, msg: Joy) -> None:
        """
        Callback function for joy topic.

        Maps joystick axes to AkmControl command.
        - Axis 2 -> velocity
        - Axis 3 -> steering_angle
        """
        # Check if axes exist
        if len(msg.axes) <= max(self.velocity_axis, self.steering_axis):
            self.get_logger().warn(
                f"Joy message does not have enough axes. "
                f"Expected at least {max(self.velocity_axis, self.steering_axis) + 1}, "
                f"got {len(msg.axes)}"
            )
            return

        # Get axis values (typically -1.0 to 1.0)
        velocity_input = msg.axes[self.velocity_axis]
        steering_input = msg.axes[self.steering_axis]

        shaped_velocity_input = self._apply_sensitivity_curve(velocity_input)

        # Map axis values to command ranges
        # velocity: -1.0 to 1.0 -> -max_velocity to max_velocity
        velocity = velocity_input * self.max_velocity
        # steering_angle: -1.0 to 1.0 -> -max_steering_angle to max_steering_angle
        steering_angle = -1 * int(shaped_steering_input * self.max_steering_angle)

        # Create and publish AkmControl message
        control_msg = AkmControl()
        control_msg.velocity = velocity
        control_msg.steering_angle = steering_angle

        self.control_pub.publish(control_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
