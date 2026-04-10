from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Launch steering_control for autonomous steering control."""

    launch_args = [
        DeclareLaunchArgument("model_path", default_value="path/to/your/model.pt", description="Path to the steering net model file"),
        DeclareLaunchArgument("publish_rate", default_value="10.0", description="Rate (in Hz) to publish control commands when enabled"),
        DeclareLaunchArgument("joy_enable_button", default_value="0", description="Button index to enable/disable the controller"),
        DeclareLaunchArgument("enable_refresh_rate", default_value="5.0", description="Rate (in Hz) to check the enable state based on the latest Joy message"),
        DeclareLaunchArgument("image_topic", default_value="/camera/camera/color/image_raw", description="Topic for input RGB images"),
        DeclareLaunchArgument("control_topic", default_value="/movement_control", description="Topic to publish control commands to"),
        DeclareLaunchArgument("joy_topic", default_value="/joy", description="Topic for joystick input to enable/disable the controller"),
        DeclareLaunchArgument("velocity", default_value="0.3", description="Constant velocity to use when publishing control commands"),
    ]

    steering_control = Node(
        package="msdc_ros",
        executable="lane_follow_node",
        parameters=[
            {"model_path": LaunchConfiguration("model_path")},
            {"publish_rate": LaunchConfiguration("publish_rate")},
            {"joy_enable_button": LaunchConfiguration("joy_enable_button")},
            {"enable_refresh_rate": LaunchConfiguration("enable_refresh_rate")},
            {"image_topic": LaunchConfiguration("image_topic")},
            {"control_topic": LaunchConfiguration("control_topic")},
            {"joy_topic": LaunchConfiguration("joy_topic")},
            {"velocity": LaunchConfiguration("velocity")},
        ],
    )

    # Joy node for joystick input
    joy_node = Node(
        package="joy",
        executable="joy_node",
    )

    # Driver node for the rosmaster
    driver_node = Node(
        package="rosmaster_r2_akm_driver",
        executable="ackman_driver_r2",
        parameters=[{"driver_sleep_time": 0.002, "state_pub_period": 1.0}],
    )

    # RealSense camera node
    realsense_node = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        parameters=[
            {"rgb_camera.color_profile": "640x480x30"},
            {"enable_depth": False},
            {"enable_infra1": False},
            {"enable_infra2": False},
        ],
    )

    return LaunchDescription([
        *launch_args,
        lane_follow_node,
        joy_node,
        driver_node,
        realsense_node,
    ])
