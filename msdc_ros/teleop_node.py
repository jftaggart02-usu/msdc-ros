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

        # Get parameter values
        joy_topic = self.get_parameter("joy_topic").value
        control_topic = self.get_parameter("control_topic").value
        self.velocity_axis = self.get_parameter("velocity_axis").value
        self.steering_axis = self.get_parameter("steering_axis").value
        self.max_velocity = self.get_parameter("max_velocity").value
        self.max_steering_angle = self.get_parameter("max_steering_angle").value

        # Create subscriber and publisher
        self.joy_sub = self.create_subscription(Joy, joy_topic, self.joy_callback, 10)

        self.control_pub = self.create_publisher(AkmControl, control_topic, 10)

        self.get_logger().info(f"Teleop node initialized: subscribing to {joy_topic}, " f"publishing to {control_topic}")

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

        # Map axis values to command ranges
        # velocity: -1.0 to 1.0 -> -max_velocity to max_velocity
        velocity = velocity_input * self.max_velocity
        # steering_angle: -1.0 to 1.0 -> -max_steering_angle to max_steering_angle
        steering_angle = -1 * int(steering_input * self.max_steering_angle)

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
