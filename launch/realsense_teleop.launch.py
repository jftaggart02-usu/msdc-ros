from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Launch joy_node and teleop_node for teleoperation."""

    realsense_node = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="realsense2_camera",
        output="screen",
    )

    # Joy node for joystick input
    joy_node = Node(
        package="joy",
        executable="joy_node",
        name="joy_node",
        output="screen",
    )

    # Teleop node to convert joy messages to AkmControl
    teleop_node = Node(
        package="msdc_ros",
        executable="teleop_node",
        name="teleop_node",
        output="screen",
        parameters=[
            {"joy_topic": "/joy"},
            {"control_topic": "/movement_control"},
            {"velocity_axis": 1},
            {"steering_axis": 2},
            {"max_velocity": 0.7},
            {"max_steering_angle": 45},
            {"joystick_sensitivity_curve": "nonlinear"},
            {"joystick_sensitivity_alpha": 0.6},
        ],
    )

    ns_arg = DeclareLaunchArgument("namespace", default_value="", description="Namespace for the node")

    # Driver node for the rosmaster
    driver_node = Node(
        package="rosmaster_r2_akm_driver",
        executable="ackman_driver_r2",
        parameters=[{"driver_sleep_time": 0.002, "state_pub_period": 1.0}],
        namespace=LaunchConfiguration("namespace"),
    )

    return LaunchDescription(
        [
            realsense_node,
            joy_node,
            teleop_node,
            ns_arg,
            driver_node,
        ]
    )
