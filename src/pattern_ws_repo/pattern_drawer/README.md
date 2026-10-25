# pattern_drawer

Dual-arm drawing setup in Gazebo (ROS 2 Humble). A Franka Panda with a pen draws a
shape on a table, then a UR5 with a wrist camera scans the drawing and checks what
was drawn. At the end you get a small report: which shape was asked for, which
shape was detected, and how close the ink is to the planned path.

Internship project.

## Setup

- One table (2.0 x 1.2 m, top at z = 0.75) in `worlds/box.sdf`
- UR5 on the left end at x = -0.70, namespace `/ur5`, camera on `tool0`
- Panda on the right end at x = 0.70, turned 180 deg to face the UR5, namespace `/panda`,
  pen mounted on the hand (`urdf/panda_pen.urdf.xacro`)
- Each arm has its own controller manager and its own MoveIt `move_group`

## How one run works

1. **Draw** (`draw_node`): the Panda draws the chosen shape. Every line is a separate
   stroke: move above the start, go down, draw a Cartesian path, lift up. With
   `lift_at_corners` the pen is lifted at every corner. After drawing, the Panda moves
   out of the camera's view.
2. **Ink** (`pen_trail`): while the pen tip touches the table, small black strips are
   spawned in Gazebo so the camera can actually see the line. The same trail is shown in
   RViz. `/panda/clear_ink` removes all of it.
3. **Scan** (`scan_drawing`): the UR5 points the camera at a few spots over the drawing,
   takes photos and projects them onto the table plane to get one top-down image
   (1 px = 1 mm).
4. **Recognise** (`shape_recognizer`): a small CNN classifies the shape, and an OpenCV
   contour check counts the corners to cross-check the CNN.
5. **Report** (`run_report`): compares the scanned ink with the planned path and writes
   `report.txt`. It also adds one line per run to `runs/summary.csv`.


Shapes: `square_circle`, `circle`, `triangle`, `polygon` (3-6 sides), `star`, `spiral`,
and `center_square` / `max_square` / `rectangle`, which take their size from the
workspace map.

## Nodes

| Executable | What it does |
|---|---|
| `draw_gui` | Window to pick a shape and size, then run draw -> scan -> recognise -> report |
| `draw_node` | Draws a shape with the Panda |
| `pen_trail` | Ink in Gazebo + RViz, service `/panda/clear_ink` |
| `table_scene` | Adds the table as a collision object to both MoveIt instances |
| `camera_frustum` | Shows the UR5 camera's field of view in RViz and Gazebo |
| `workspace_map` | Checks with IK where the Panda can reach the table, saves `~/panda_workspace.yaml` |
| `scan_drawing` | UR5 scan, builds `scan_topdown.png` |
| `camera_view` | Older single-photo version of the scan |
| `train_shape_cnn` | Trains the CNN on generated images, saves `~/pattern_models/shape_cnn.pt` |
| `shape_recognizer` | Detects the shape in the latest scan |
| `run_report` | Writes the report for the latest run |

## Dependencies

- ROS 2 Humble, Ignition Gazebo (Fortress), `ros_gz_sim`, `ros_gz_bridge`
- MoveIt 2, `ros2_control`, `gz_ros2_control`
- `ur_description`, `ur_moveit_config` (Universal Robots)
- `franka_description`, `franka_gazebo_bringup` (franka_ros2)
- Python: `opencv`, `numpy`, `pyyaml`, `torch`, `tkinter`, `cv_bridge`

## Build


cd ~/dual_arm_ws
colcon build --packages-select pattern_drawer
source install/setup.bash

Before the first run:

- Train the CNN once: `ros2 run pattern_drawer train_shape_cnn`. This takes a few
  minutes on a CPU.
- For the `center_square` / `max_square` / `rectangle` shapes, run
  `ros2 run pattern_drawer workspace_map --ros-args -p use_sim_time:=true` once while
  the simulation is running.

## Launch

**Terminal 1**: stop anything left over from a previous run, then start the simulation:


pkill -f "ign gazebo"; pkill -f gz_ros2_control; pkill -f ros2_control_node
pkill -f move_group; pkill -f rviz2; pkill -f robot_state_publisher
pkill -f spawner; pkill -f pen_trail; pkill -f camera_frustum; pkill -f table_scene
ros2 daemon stop
sleep 3
ps aux | grep -E "ign gazebo|move_group|rviz2" | grep -v grep     # must print nothing
source ~/dual_arm_ws/install/setup.bash
ros2 launch pattern_drawer dual_arm_sim.launch.py

Wait until both MoveIt instances are up. The UR5 starts after about 8 s and the Panda
after about 10 s.

**Terminal 2**: open the GUI:

source ~/dual_arm_ws/install/setup.bash
ros2 run pattern_drawer draw_gui


In the GUI, pick a shape on the left and enter the values (or press **Random**), then
press **Draw**. The max values are limited to what the Panda can reach. The Draw button
stays disabled while a value is out of range. With both checkboxes on, the old ink is
cleared first, and scan, recognise and report run after the drawing. The output shows
in the log box at the bottom, and the report is in `runs/latest/report.txt`.

**Clear** only removes the ink. **Stop** stops the current step.

## Credits

- The UR5 launch files are adapted from
  [Universal_Robots_ROS2_Gazebo_Simulation](https://github.com/UniversalRobots/Universal_Robots_ROS2_Gazebo_Simulation)
  and `ur_moveit_config`.
- The Panda launch files are adapted from
  [franka_ros2](https://github.com/frankaemika/franka_ros2) and the MoveIt
  `panda_moveit_config` demo.
- Parts of the code were written with help from an AI assistant. I reviewed, tested
  and integrated them.
