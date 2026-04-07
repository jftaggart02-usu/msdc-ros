from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition


def generate_launch_description():
    """Launch joy_node and teleop_node for teleoperation."""

    launch_args = [
        DeclareLaunchArgument("launch_realsense", default_value="true", description="Whether to launch the RealSense camera node"),
        DeclareLaunchArgument("joy_topic", default_value="/joy", description="Topic for joystick input"),
        DeclareLaunchArgument("control_topic", default_value="/movement_control", description="Topic for movement control commands"),
        DeclareLaunchArgument("velocity_axis", default_value="1", description="Joystick axis index for velocity control"),
        DeclareLaunchArgument("steering_axis", default_value="2", description="Joystick axis index for steering control"),
        DeclareLaunchArgument("max_velocity", default_value="0.3", description="Maximum velocity for teleoperation"),
        DeclareLaunchArgument("max_steering_angle", default_value="45", description="Maximum steering angle for teleoperation"),
        DeclareLaunchArgument("steering_sensitivity_param", default_value="0.5", description="Parameter for steering sensitivity curve. Values range from 1.0 (pure cubic) to 0.0 (pure linear).")
    ]

    realsense_node = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        condition=IfCondition(LaunchConfiguration("launch_realsense")),
        parameters=[
            {"align_depth.enable": True},
            {"enable_sync": True},
            {"rgb_camera.color_profile": "640x480x30"},
            {"depth_camera.depth_profile": "640x480x30"},
            {"depth_camera.infra_profile": "640x480x30"},
        ],
    )

    # Joy node for joystick input
    joy_node = Node(
        package="joy",
        executable="joy_node",
    )

    # Teleop node to convert joy messages to AkmControl
    teleop_node = Node(
        package="msdc_ros",
        executable="teleop_node",
        parameters=[
            {"joy_topic": LaunchConfiguration("joy_topic")},
            {"control_topic": LaunchConfiguration("control_topic")},
            {"velocity_axis": LaunchConfiguration("velocity_axis")},
            {"steering_axis": LaunchConfiguration("steering_axis")},
            {"max_velocity": LaunchConfiguration("max_velocity")},
            {"max_steering_angle": LaunchConfiguration("max_steering_angle")},
            {"steering_sensitivity_param": LaunchConfiguration("steering_sensitivity_param")},
        ],
    )

    # Driver node for the rosmaster
    driver_node = Node(
        package="rosmaster_r2_akm_driver",
        executable="ackman_driver_r2",
        parameters=[{"driver_sleep_time": 0.002, "state_pub_period": 1.0}],
    )

    return LaunchDescription(
        [
            *launch_args,
            realsense_node,
            joy_node,
            teleop_node,
            driver_node,
        ]
    )