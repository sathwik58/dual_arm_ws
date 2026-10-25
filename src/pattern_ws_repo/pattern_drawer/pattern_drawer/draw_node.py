"""Draw shapes on the table with the Panda's pen.

Run (with the simulation and the Panda's MoveIt running):
  ros2 run pattern_drawer draw_node --ros-args -p use_sim_time:=true -p shape:=center_square

Shapes (parameter 'shape'):
  center_square : largest square centred in front of the Panda (from workspace_map)
  max_square    : largest reachable square anywhere          (from workspace_map)
  rectangle     : largest reachable rectangle                (from workspace_map)
  square_circle : square with a circle inside (center_x/y, square_size, circle_radius)
  circle        : circle of circle_radius around center_x/y
  triangle      : triangle, corners on circle_radius
  polygon       : regular polygon with 'sides' corners on circle_radius
  star          : 5-point star, outer points on circle_radius
  spiral        : 3-turn spiral out to circle_radius

How every line is drawn (one "stroke"):
  1. pen is up (LIFT above the table)
  2. move in the air to above the start of the line   - any re-orientation happens here
  3. go straight down until the pen touches the table
  4. draw the line (Cartesian path, pen on the table)
  5. go straight up again
With lift_at_corners:=true (default) every side of a square is its own stroke, so the pen
is lifted at each corner and never turns or re-configures while touching the table.
"""
import math
import os
import time
import rclpy
import yaml
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (Constraints, MotionPlanRequest, OrientationConstraint,
                             PositionConstraint)
from moveit_msgs.srv import GetCartesianPath
from rclpy.action import ActionClient
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive

NS = '/panda'
GROUP = 'panda_fer_arm'
TCP = 'panda_fer_hand_tcp'      # MoveIt plans for this link; the pen tip is below it
FRAME = 'world'

TABLE_Z = 0.75                  # real table top (Gazebo)
GAP = 0.005                     # tcp -> start of pen   (panda_pen.urdf.xacro)
PEN_LEN = 0.10                  # pen length            (panda_pen.urdf.xacro)
PRESS = 0.001                   # pen tip this far above the table while drawing
LIFT = 0.05                     # pen-up height above the table
PARK = (0.60, 0.0, 1.20)        # tcp position after drawing: high above the Panda's own base

SUCCESS = 1                     # moveit_msgs/MoveItErrorCodes.SUCCESS
RUNS = os.path.expanduser('~/dual_arm_ws/src/pattern_ws_repo/pattern_drawer/runs')
CLASS_OF = {'center_square': 'square', 'max_square': 'square', 'rectangle': 'square',
            'square_circle': 'square_circle', 'circle': 'circle', 'triangle': 'triangle',
            'star': 'star', 'spiral': 'spiral'}


class DrawNode(Node):
    def __init__(self):
        super().__init__('draw_node')

        self.declare_parameter('shape', 'center_square')
        self.declare_parameter('workspace_file', os.path.expanduser('~/panda_workspace.yaml'))
        self.declare_parameter('margin', 0.02)          # stay this far inside the mapped area (m)
        self.declare_parameter('lift_at_corners', True)  # lift the pen between the sides
        self.declare_parameter('park', True)             # move the Panda away after drawing

        # square_circle parameters
        self.declare_parameter('center_x', 0.30)
        self.declare_parameter('center_y', 0.0)
        self.declare_parameter('square_size', 0.16)
        self.declare_parameter('circle_radius', 0.06)
        self.declare_parameter('circle_points', 72)
        self.declare_parameter('sides', 6)              # for shape:=polygon

        self.declare_parameter('yaw', 0.0)              # pen rotation about vertical (rad)
        self.declare_parameter('speed_scale', 0.5)      # 0.5 = half the planned speed
        self.declare_parameter('runs_dir', RUNS)
        self.cartesian = self.create_client(GetCartesianPath, f'{NS}/compute_cartesian_path')
        self.move = ActionClient(self, MoveGroup, f'{NS}/move_action')
        self.execute = ActionClient(self, ExecuteTrajectory, f'{NS}/execute_trajectory')

    # ------------------------------------------------------------ helpers
    def p(self, name):
        return self.get_parameter(name).value

    def tcp_pose(self, x, y, tip_height):
        """tcp pose so that the pen tip is at (x, y, TABLE_Z + tip_height), pointing down."""
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = TABLE_Z + tip_height + GAP + PEN_LEN
        yaw = self.p('yaw')
        # 180 deg about x (z axis points down), then turned by yaw about the vertical
        pose.orientation.x = math.cos(yaw / 2)
        pose.orientation.y = math.sin(yaw / 2)
        pose.orientation.z = 0.0
        pose.orientation.w = 0.0
        return pose

    def wait(self, future, timeout=60.0):
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return future.result()

    # ------------------------------------------------------------ shapes
    def mapped_box(self, key):
        """Corners (closed outline) of a box saved by workspace_map, shrunk by 'margin'."""
        path = self.p('workspace_file')
        try:
            with open(path) as f:
                box = yaml.safe_load(f)[key]
        except (OSError, KeyError, TypeError):
            self.get_logger().error(f"No '{key}' in {path} - run workspace_map first")
            return None
        m = self.p('margin')
        x0, x1 = box['x_min'] + m, box['x_max'] - m
        y0, y1 = box['y_min'] + m, box['y_max'] - m
        if x1 <= x0 or y1 <= y0:
            self.get_logger().error(f"'{key}' is smaller than 2 x margin")
            return None
        self.get_logger().info(
            f'{key}: x {x0:.3f}..{x1:.3f}, y {y0:.3f}..{y1:.3f} ({x1 - x0:.3f} x {y1 - y0:.3f} m)')
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]

    def square(self):
        cx, cy, h = self.p('center_x'), self.p('center_y'), self.p('square_size') / 2
        return [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h),
                (cx - h, cy + h), (cx - h, cy - h)]

    def circle(self):
        cx, cy, r, n = (self.p('center_x'), self.p('center_y'), self.p('circle_radius'),
                        self.p('circle_points'))
        return [(cx + r * math.cos(2 * math.pi * i / n),
                 cy + r * math.sin(2 * math.pi * i / n)) for i in range(n + 1)]

    def regular_polygon(self, n, radius, rotation=math.pi / 2):
        """Closed outline of a regular n-gon around (center_x, center_y)."""
        cx, cy = self.p('center_x'), self.p('center_y')
        pts = [(cx + radius * math.cos(rotation + 2 * math.pi * i / n),
                cy + radius * math.sin(rotation + 2 * math.pi * i / n)) for i in range(n)]
        return pts + [pts[0]]

    def star(self, radius, spikes=5):
        """Closed star outline: points alternate between radius and 0.4 * radius."""
        cx, cy = self.p('center_x'), self.p('center_y')
        pts = []
        for i in range(2 * spikes):
            r = radius if i % 2 == 0 else radius * 0.4
            a = math.pi / 2 + i * math.pi / spikes
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        return pts + [pts[0]]

    def spiral(self, radius, turns=3):
        cx, cy, n = self.p('center_x'), self.p('center_y'), self.p('circle_points')
        total = n * turns
        return [(cx + radius * i / total * math.cos(2 * math.pi * turns * i / total),
                 cy + radius * i / total * math.sin(2 * math.pi * turns * i / total))
                for i in range(total + 1)]

    def polygon_strokes(self, corners, name):
        """One stroke per side (pen lifted at every corner), or the whole outline at once."""
        if not self.p('lift_at_corners'):
            return [(corners, name)]
        return [([corners[i], corners[i + 1]], f'{name} side {i + 1}')
                for i in range(len(corners) - 1)]

    # ------------------------------------------------------------ motions
    def move_to(self, pose):
        """Free-space move (MoveGroup / OMPL) of the tcp to pose. Used in the air only."""
        region = SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.005])
        pos = PositionConstraint()
        pos.header.frame_id = FRAME
        pos.link_name = TCP
        pos.constraint_region.primitives = [region]
        pos.constraint_region.primitive_poses = [pose]
        pos.weight = 1.0

        ori = OrientationConstraint()
        ori.header.frame_id = FRAME
        ori.link_name = TCP
        ori.orientation = pose.orientation
        ori.absolute_x_axis_tolerance = 0.02
        ori.absolute_y_axis_tolerance = 0.02
        ori.absolute_z_axis_tolerance = 0.02
        ori.weight = 1.0

        req = MotionPlanRequest()
        req.group_name = GROUP
        req.num_planning_attempts = 5
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = 0.2
        req.max_acceleration_scaling_factor = 0.2
        req.start_state.is_diff = True
        req.goal_constraints = [Constraints(position_constraints=[pos],
                                            orientation_constraints=[ori])]

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options.plan_only = False

        handle = self.wait(self.move.send_goal_async(goal))
        if handle is None or not handle.accepted:
            self.get_logger().error('MoveGroup goal was rejected')
            return False
        result = self.wait(handle.get_result_async(), timeout=120.0)
        code = result.result.error_code.val if result else None
        if code != SUCCESS:
            self.get_logger().error(f'Free-space move failed (error code {code})')
            return False
        return True

    def slow_down(self, traj, scale):
        """Stretch a trajectory in time: scale 0.5 -> takes twice as long."""
        if scale >= 1.0:
            return traj
        for pt in traj.joint_trajectory.points:
            t = (pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9) / scale
            pt.time_from_start = Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
            pt.velocities = [v * scale for v in pt.velocities]
            pt.accelerations = [a * scale * scale for a in pt.accelerations]
        return traj

    def cartesian_move(self, poses, name, quiet=False):
        """Straight-line moves of the tcp through poses, then execute. Returns True on success."""
        req = GetCartesianPath.Request()
        req.header.frame_id = FRAME
        req.group_name = GROUP
        req.link_name = TCP
        req.waypoints = poses
        req.max_step = 0.005          # interpolate every 5 mm
        req.jump_threshold = 0.0      # 0 = no joint-jump check
        req.avoid_collisions = True
        req.start_state.is_diff = True

        res = self.wait(self.cartesian.call_async(req))
        if res is None or res.error_code.val != SUCCESS:
            if not quiet:
                self.get_logger().error(f'{name}: Cartesian path service failed')
            return False
        if res.fraction < 0.99:
            if not quiet:
                self.get_logger().error(
                    f'{name}: only {res.fraction * 100:.0f}% of the path is reachable - '
                    f'make the shape smaller (larger margin) or try another yaw')
            return False

        traj = self.slow_down(res.solution, self.p('speed_scale'))
        handle = self.wait(self.execute.send_goal_async(ExecuteTrajectory.Goal(trajectory=traj)))
        if handle is None or not handle.accepted:
            self.get_logger().error(f'{name}: execution was rejected')
            return False
        result = self.wait(handle.get_result_async(), timeout=180.0)
        code = result.result.error_code.val if result else None
        if code != SUCCESS:
            self.get_logger().error(f'{name}: execution failed (error code {code})')
            return False
        return True

    def go_above(self, x, y, name):
        """Pen up, move in the air to above (x, y). Straight line first; if that is not
        possible (arm must re-configure), let the planner find a path - still in the air."""
        above = self.tcp_pose(x, y, LIFT)
        if self.cartesian_move([above], f'{name} (travel)', quiet=True):
            return True
        self.get_logger().info(f'{name}: re-positioning the arm in the air')
        return self.move_to(above)

    def stroke(self, points, name):
        """Draw one line: above start -> pen down -> follow points -> pen up."""
        x0, y0 = points[0]
        xn, yn = points[-1]
        if not self.go_above(x0, y0, name):
            return False
        poses = [self.tcp_pose(x0, y0, PRESS)]                     # straight down
        poses += [self.tcp_pose(x, y, PRESS) for x, y in points[1:]]  # draw
        poses.append(self.tcp_pose(xn, yn, LIFT))                   # straight up
        if not self.cartesian_move(poses, name):
            return False
        self.get_logger().info(f'{name}: done')
        return True

    # ------------------------------------------------------------ main sequence
    def run(self):
        shape = self.p('shape')
        if shape == 'square_circle':
            if self.p('circle_radius') >= self.p('square_size') / 2:
                self.get_logger().error('circle_radius must be smaller than square_size / 2')
                return
            strokes = self.polygon_strokes(self.square(), 'Square') + [(self.circle(), 'Circle')]
        elif shape == 'circle':
            strokes = [(self.circle(), 'Circle')]
        elif shape == 'triangle':
            strokes = self.polygon_strokes(self.regular_polygon(3, self.p('circle_radius')),
                                           'Triangle')
        elif shape == 'polygon':
            n = self.p('sides')
            strokes = self.polygon_strokes(self.regular_polygon(n, self.p('circle_radius')),
                                           f'{n}-gon')
        elif shape == 'star':
            strokes = self.polygon_strokes(self.star(self.p('circle_radius')), 'Star')
        elif shape == 'spiral':
            strokes = [(self.spiral(self.p('circle_radius')), 'Spiral')]
        elif shape in ('center_square', 'max_square', 'rectangle'):
            key = {'center_square': 'center_square', 'max_square': 'square',
                   'rectangle': 'rectangle'}[shape]
            corners = self.mapped_box(key)
            if corners is None:
                return
            strokes = self.polygon_strokes(corners, shape)
        else:
            self.get_logger().error(
                "shape must be one of: center_square, max_square, rectangle, square_circle, "
                "circle, triangle, polygon, star, spiral")
            return

        self.get_logger().info('Waiting for the Panda MoveIt services ...')
        self.cartesian.wait_for_service()
        self.move.wait_for_server()
        self.execute.wait_for_server()

        # First approach: always a planned free-space move, pen up
        x0, y0 = strokes[0][0][0]
        self.get_logger().info('Moving above the start point (pen up)')
        if not self.move_to(self.tcp_pose(x0, y0, LIFT)):
            return

        for points, name in strokes:
            if not self.stroke(points, name):
                return
        self.get_logger().info(f'Drawing finished: {shape}')
        self.save_last_drawing(strokes, shape)
        self.start_run(strokes, shape)
        if self.p('park'):
            self.get_logger().info('Parking the Panda out of the camera view')
            self.park()

    def park(self):
        """Move the pen high above the Panda's own base, away from the drawing."""
        pose = self.tcp_pose(PARK[0], PARK[1], 0.0)
        pose.position.z = PARK[2]
        if self.move_to(pose):
            self.get_logger().info('Panda parked')

    def save_last_drawing(self, strokes, shape):
        """Store the bounding box of what was drawn, so camera_view can look at it."""
        xs = [x for points, _ in strokes for x, _ in points]
        ys = [y for points, _ in strokes for _, y in points]
        path = self.p('workspace_file')
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except OSError:
            data = {}
        data['last_drawing'] = {'shape': shape,
                                'x_min': round(min(xs), 4), 'x_max': round(max(xs), 4),
                                'y_min': round(min(ys), 4), 'y_max': round(max(ys), 4)}
        with open(path, 'w') as f:
            yaml.safe_dump(data, f, sort_keys=False)
        self.get_logger().info(f'Saved the drawing area to {path} (key last_drawing)')

    def start_run(self, strokes, shape):
        """New dated folder for this run + meta.yaml (what was asked)."""
        runs = self.p('runs_dir')
        name = time.strftime('%Y-%m-%d_%H-%M-%S') + '_' + shape
        os.makedirs(os.path.join(runs, name), exist_ok=True)
        latest = os.path.join(runs, 'latest')
        if os.path.islink(latest):
            os.remove(latest)
        os.symlink(name, latest)
        if shape == 'polygon':
            expected = {3: 'triangle', 4: 'square', 5: 'pentagon', 6: 'hexagon'}[self.p('sides')]
        else:
            expected = CLASS_OF[shape]
        meta = {'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'shape': shape, 'expected_class': expected,
                'params': {k: float(self.p(k)) for k in
                           ('center_x', 'center_y', 'square_size', 'circle_radius', 'sides')},
                'strokes': [[[round(float(x), 4), round(float(y), 4)] for x, y in pts]
                            for pts, _ in strokes]}
        with open(os.path.join(runs, name, 'meta.yaml'), 'w') as f:
            yaml.safe_dump(meta, f, sort_keys=False)
        self.get_logger().info(f'Run folder: {os.path.join(runs, name)}')


def main():
    rclpy.init()
    node = DrawNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()