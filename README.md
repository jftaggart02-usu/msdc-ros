# MSDC (Mini Self-Driving Car)

## Prerequisites

You must have a Rosmaster R2L robot with the regular Jetson Nano swapped for a Jetson Orin Nano. Additionally, the Jetson Orin Nano should already be set up with JetPack 6.2.1 and have ROS2 Humble installed.

Also, the following APT packages should be installed:
```
sudo apt install ros-humble-joy
sudo apt-get install joystick
```

## Setup UDEV rules and Install USB Driver

Create a file called `usb.rules` inside `/etc/udev/rules.d`. Paste this into the file:
```
KERNEL=="ttyUSB*", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", MODE:="0777", SYMLINK+="myserial"
```

Then load the UDEV rules:
```
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Then plug in the Rosmaster expansion board, run `lsusb`, and you should see:
```
Bus 001 Device 024: ID 1a86:7523 QinHeng Electronics CH340 serial converter
```

The Jetson Orin Nano doesn't come with the CH341 driver already installed. Follow the tutorial at [this link](https://www.youtube.com/watch?v=RHqSR3Wj_K0) to install the CH341 driver. Here is a summary:
- `cd` to a directory of your choice and run `git clone https://github.com/jetsonhacks/jetson-orin-kernel-builder.git`.
- `cd jetson-orin-kernel-builder/prebuilt/jetpack-6.2.1`
- Check to make sure we have a good download: `shasum -c usb_serial_ch341.tar.xz.sha256`
- Then open a file browser and extract `usb_serial_ch341.tar.xz`.
- `cd usb_serial_ch341`
- Install the driver: `./install_module_ch341.sh`
- Replug the rosmaster expansion board
- Check if board is recognized with `ls /dev/ttyUSB*`. If nothing shows up, proceed to next step.
- Quick fix: `sudo apt purge brltty`
- Now, the device should show up: `ls /dev/ttyUSB*`
- Additionally, the `myserial` symlink should show up when you run `ls /dev`.

## Install Realsense SDK and ROS Wrapper

**Important**: these instructions must be followed for installing Realsense SDK on Jetson Orin Nano! Following the instructions on the official Realsense website doesn't work because the Jetson Orin Nano uses a modified version of Ubuntu.

Follow the instructions in the README in [this repository](https://github.com/jetsonhacks/jetson-orin-librealsense).

After you follow these instructions and run `realsense-viewer`, you may get the following error:
```
Multiple realsense udev-rules were found! : 1:/etc/udev/rules.d/99-realsense-libusb.rules 2: /lib/udev/rules.d/60-librealsense2-udev-rules.rules Make sure to remove redundancies!
```

If you do, simply delete the UDEV rules in the `/etc` directory:
```
sudo rm /etc/udev/rules.d/99-realsense-libusb.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Now, install the ROS2 wrapper for the Realsense SDK:
```
sudo apt install ros-humble-realsense2-*
```

## Install Rosmaster Library

```
cd /opt
sudo git clone https://github.com/jftaggart02/Rosmaster_Lib.git
cd Rosmaster_Lib
sudo python3 setup.py install
```

## Install the Software

Add the following aliases to `~/.bashrc`:
```
alias make_venv='python3 -m venv --system-site-packages venv && touch venv/COLCON_IGNORE && source venv/bin/activate && rosdep install --from-paths src --ignore-src -r -y'
alias sd='source /opt/ros/humble/setup.bash && source venv/bin/activate && . install/setup.bash'
alias build='source /opt/ros/humble/setup.bash && source venv/bin/activate && python3 -m colcon build && . install/local_setup.bash'
```

Install the workspace and create a virtual environment. Note, you must have SSH key access to the `droge-robotics` gitlab group to install `rosmaster_r2_akm_driver`.
```
mkdir -p msdc_ws/src
cd msdc_ws/src
git clone https://github.com/jftaggart02-usu/msdc-ros
git clone https://github.com/jftaggart02-usu/msdc-core
git clone git@gitlab.com:utahstate/droge-robotics/general_research_code/rosmaster_r2_akm_driver.git
cd ..
make_venv
pip install -r ./src/msdc-ros/requirements.txt
pip install -e ./src/msdc-core
build
```

Whenever you open a new terminal, run the following command (from the msdc_ws directory) to source the workspace and activate the virtual environment.
```
sd
```

## Joystick Test

Plug in your joystick to a USB port and run the following command to check if it is recognized by the system. You should see `js0` listed.
```
ls /dev/input
```

Test joystick using the following command. You should see values for each of the axes and buttons printed to the screen. Verify the values change when you press the buttons and move the axes.
```
sudo jstest /dev/input/js0
```

Test joystick with ROS2.
```
ros2 run joy joy_node
ros2 topic echo /joy  # in another terminal
```

In the terminal where you ran `ros2 topic echo /joy`, you should see messages containing the axis and button values. They should change when you press the buttons and move the axes.

## Run Teleop Node

Make sure you have the workspace built with `build` (one-time setup) and sourced with `sd` (every time you start a new terminal). Then run the launch file:
```
ros2 launch msdc_ros teleop.launch.py
```

The robot should follow your commands!

## Using Rosbag to Record Camera Feed and Steering Angle

First, run `ros2 launch msdc_ros realsense_teleop.launch.py` and ensure the teleop is working properly. 

Then, run the following command, but replace `[output_dir]` with a path to the desired output directory.
```
ros2 bag record -o [output_dir] /movement_control /camera/realsense2_camera/color/image_raw
```

Drive the car around, and when finished, hit `ctrl-c` in the terminal where you ran the rosbag record script.

## Processing the Rosbag to create Dataset

This package has a script called `process_rosbag.py` that can process a Rosbag and create a dataset file structure that can be used to train the steering control network. You can type `ros2 run msdc_ros process_rosbag --help` to see a full list of arguments. For example,
```
ros2 run msdc_ros process_rosbag --image-topic /camera/realsense2_camera/color/image_raw --control-topic /movement_control ./rosbag/rosbag_test_1 ./datasets/dataset_test_1
```
