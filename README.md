# dual_arm_ws

ROS 2 Humble workspace for a dual-arm drawing simulation in Ignition Gazebo.
A Franka Panda draws shapes on a table with a pen, and a UR5 with a wrist camera scans
the drawing, recognises the shape and writes a report on how accurate it was.

Demo video: [docs/demo.webm](docs/demo.webm)

My own package is `src/pattern_ws_repo/pattern_drawer`. Its README has the details
and the launch steps. The other folders in `src/` are the upstream robot packages it
depends on, added as git submodules.

## Contents

| Folder | What it is |
|---|---|
| `src/pattern_ws_repo/pattern_drawer` | My package: world, launch files, drawing, scanning, CNN shape recognition, GUI |
| `src/franka_ros2` | Franka ROS 2 packages (Gazebo bringup, MoveIt config) |
| `src/franka_description` | Franka robot models and meshes |
| `src/libfranka` | Franka C++ library (needed by franka_ros2) |
| `src/Universal_Robots_ROS2_Gazebo_Simulation` | UR Gazebo simulation (reference only, not built) |
| `src/olvx_descriptions_module` | Olive Robotics description models |

## Clone and build

```bash
git clone --recurse-submodules https://github.com/sathwik58/dual_arm_ws.git
cd dual_arm_ws
touch src/Universal_Robots_ROS2_Gazebo_Simulation/COLCON_IGNORE   # not built
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

The UR description and MoveIt config come from the ROS packages
(`sudo apt install ros-humble-ur-description ros-humble-ur-moveit-config`).

## Run

Terminal 1:

```bash
source ~/dual_arm_ws/install/setup.bash
ros2 launch pattern_drawer dual_arm_sim.launch.py
```

Terminal 2:

```bash
source ~/dual_arm_ws/install/setup.bash
ros2 run pattern_drawer draw_gui
```
