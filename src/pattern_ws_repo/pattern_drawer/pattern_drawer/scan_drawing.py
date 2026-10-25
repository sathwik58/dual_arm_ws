"""Scan the drawing area with the UR5 wrist camera and build one top-down image of it.

Run after draw_node (which saves 'last_drawing' and parks the Panda):
  ros2 run pattern_drawer scan_drawing --ros-args -p use_sim_time:=true

Steps
  1. Read the drawing area ('last_drawing') from ~/panda_workspace.yaml, add a margin.
  2. Choose target points over the area (centre, left, right, and near/far if it is large).
  3. For each target: move the UR5 so the camera looks at it (angled, from the UR5's side),
     save the photo + camera pose + lens data (view_N.png / view_N.json).
  4. Project every photo onto the table plane (z = 0.75) -> top-down view, 1 px = 1 mm.
  5. Combine them (darkest pixel wins = ink) -> scan_topdown.png
"""
import json
import math
import os
import time

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (Constraints, MotionPlanRequest, OrientationConstraint,
                             PositionConstraint)
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener

NS = '/ur5'
GROUP = 'ur5_ur_manipulator'
TOOL = 'ur5_tool0'
FRAME = 'world'
OPTICAL = 'ur5_camera_optical_frame'
UR5_BASE = (-0.70, 0.0, 0.75)
TABLE_Z = 0.75
CAM_OFFSET = 0.01               # camera sits 1 cm in front of tool0 along tool0 z
SUCCESS = 1
RUNS = os.path.expanduser('~/dual_arm_ws/src/pattern_ws_repo/pattern_drawer/runs')


# ---------------------------------------------------------------- small maths helpers
def norm(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def matrix_to_quaternion(m):
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return ((m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, 0.25 * s)
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        return (0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s)
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        return ((m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s)
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
    return ((m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s, (m[1][0] - m[0][1]) / s)


def look_at_quaternion(forward):
    """Orientation whose z axis points along 'forward' (tool0 z = camera viewing direction)."""
    z = norm(forward)
    x = norm(cross((0.0, 0.0, 1.0), z))
    y = cross(z, x)
    return matrix_to_quaternion([[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]])


def quaternion_to_matrix(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def topdown_from_view(img, K, R, t, x_range, y_range, mm=1.0):
    """Project one photo onto the table plane. Rows = world x (near -> far),
    columns = world y. Returns (top-down image, mask of pixels this view actually saw)."""
    xs = np.arange(x_range[0], x_range[1], mm / 1000.0)
    ys = np.arange(y_range[0], y_range[1], mm / 1000.0)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    P = np.stack([X, Y, np.full_like(X, TABLE_Z)], axis=-1)    # table points in world
    Pc = (P - np.asarray(t)) @ np.asarray(R)                    # = R^T (P - t): camera frame
    front = Pc[..., 2] > 1e-6
    z = np.where(front, Pc[..., 2], 1.0)
    u = K[0][0] * Pc[..., 0] / z + K[0][2]
    v = K[1][1] * Pc[..., 1] / z + K[1][2]
    h, w = img.shape[:2]
    seen = front & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    top = cv2.remap(img, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    top[~seen] = 255
    return top, seen


class ScanDrawing(Node):
    def __init__(self):
        super().__init__('scan_drawing')
        self.declare_parameter('workspace_file', os.path.expanduser('~/panda_workspace.yaml'))
        self.declare_parameter('area', 'last_drawing')
        self.declare_parameter('margin', 0.05)          # scan this much around the drawing (m)
        self.declare_parameter('run_dir', os.path.join(RUNS, 'latest'))
        self.declare_parameter('mm_per_pixel', 1.0)

        self.move = ActionClient(self, MoveGroup, f'{NS}/move_action')
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.bridge = CvBridge()
        self.image, self.info = None, None
        self.create_subscription(Image, '/ur5/camera/image_raw', self.on_image, 1)
        self.create_subscription(CameraInfo, '/ur5/camera/camera_info', self.on_info, 1)

    def on_image(self, msg):
        self.image = msg

    def on_info(self, msg):
        self.info = msg

    def wait(self, future, timeout=60.0):
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return future.result()

    def spin_for(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    # ------------------------------------------------------------ area and targets
    def scan_area(self):
        path = self.get_parameter('workspace_file').value
        key = self.get_parameter('area').value
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            b = data[key]
        except (OSError, KeyError, TypeError):
            self.get_logger().error(f"No '{key}' in {path} - run draw_node first")
            return None
        m = self.get_parameter('margin').value
        return b['x_min'] - m, b['x_max'] + m, b['y_min'] - m, b['y_max'] + m

    @staticmethod
    def targets(x0, x1, y0, y1):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = y1 - y0, x1 - x0
        pts = [(cx, cy), (cx, y0 + w / 4), (cx, y1 - w / 4)]          # centre, left, right
        if h > 0.30:
            pts += [(x0 + h / 4, cy), (x1 - h / 4, cy)]                # near, far
        return pts

    # ------------------------------------------------------------ UR5 motion
    def move_tool(self, pos, quat):
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = pos
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
        pc = PositionConstraint()
        pc.header.frame_id = FRAME
        pc.link_name = TOOL
        pc.constraint_region.primitives = [SolidPrimitive(type=SolidPrimitive.SPHERE,
                                                          dimensions=[0.01])]
        pc.constraint_region.primitive_poses = [pose]
        pc.weight = 1.0
        oc = OrientationConstraint()
        oc.header.frame_id = FRAME
        oc.link_name = TOOL
        oc.orientation = pose.orientation
        oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = 0.05
        oc.absolute_z_axis_tolerance = 0.05
        oc.weight = 1.0
        req = MotionPlanRequest()
        req.group_name = GROUP
        req.num_planning_attempts = 3
        req.allowed_planning_time = 3.0
        req.max_velocity_scaling_factor = 0.2
        req.max_acceleration_scaling_factor = 0.2
        req.start_state.is_diff = True
        req.goal_constraints = [Constraints(position_constraints=[pc],
                                            orientation_constraints=[oc])]
        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options.plan_only = False
        handle = self.wait(self.move.send_goal_async(goal))
        if handle is None or not handle.accepted:
            return False
        res = self.wait(handle.get_result_async(), timeout=60.0)
        code = res.result.error_code.val if res is not None else 'no result'
        if code != SUCCESS:
            self.get_logger().info(f'  MoveIt error code: {code}', throttle_duration_sec=2.0)
        return code == SUCCESS

    def look_at(self, target):
        """Try camera spots between the target and the UR5 base until one is reachable."""
        tgt = (target[0], target[1], TABLE_Z)
        for back in (0.45, 0.55, 0.65, 0.75):
            for height in (0.50, 0.40, 0.60, 0.30):
                cam = (tgt[0] + (UR5_BASE[0] - tgt[0]) * back,
                       tgt[1] + (UR5_BASE[1] - tgt[1]) * back,
                       TABLE_Z + height)
                fwd = norm(tuple(a - b for a, b in zip(tgt, cam)))
                tool = tuple(c - CAM_OFFSET * f for c, f in zip(cam, fwd))
                if self.move_tool(tool, look_at_quaternion(fwd)):
                    return True
        return False

    # ------------------------------------------------------------ capture
    def capture(self, index, folder):
        """Fresh image + camera pose + K after the arm has stopped."""
        self.spin_for(1.5)                     # let the arm settle and new images arrive
        self.image = None
        end = time.time() + 5.0
        while (self.image is None or self.info is None) and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.image is None or self.info is None:
            self.get_logger().warn('No camera image / camera_info received')
            return None
        try:
            tf = self.tf.lookup_transform(FRAME, OPTICAL, Time()).transform
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'No camera pose from TF: {e}')
            return None
        img = self.bridge.imgmsg_to_cv2(self.image, desired_encoding='bgr8')
        q = (tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w)
        view = {'K': np.array(self.info.k).reshape(3, 3).tolist(),
                'R': quaternion_to_matrix(q).tolist(),
                't': [tf.translation.x, tf.translation.y, tf.translation.z]}
        cv2.imwrite(os.path.join(folder, f'view_{index}.png'), img)
        with open(os.path.join(folder, f'view_{index}.json'), 'w') as f:
            json.dump(view, f, indent=2)
        return img, view

    # ------------------------------------------------------------ main
    def run(self):
        area = self.scan_area()
        if area is None:
            return
        x0, x1, y0, y1 = area
        folder = os.path.realpath(self.get_parameter('run_dir').value)
        if not os.path.isdir(folder):
            self.get_logger().error(f'{folder} not found - run draw_node first')
            return
        views_dir = os.path.join(folder, 'views')
        os.makedirs(views_dir, exist_ok=True)
        self.get_logger().info(f'Scanning x {x0:.2f}..{x1:.2f}, y {y0:.2f}..{y1:.2f}')
        self.move.wait_for_server()

        views = []
        for i, tgt in enumerate(self.targets(x0, x1, y0, y1), start=1):
            self.get_logger().info(f'View {i}: looking at ({tgt[0]:.2f}, {tgt[1]:.2f})')
            if not self.look_at(tgt):
                self.get_logger().warn(f'View {i}: not reachable, skipped')
                continue
            shot = self.capture(i, views_dir)
            if shot:
                views.append(shot)
                self.get_logger().info(f'View {i}: photo saved')

        if not views:
            self.get_logger().error('No views captured - can the UR5 plan (Plan & Execute)?')
            return

        mm = self.get_parameter('mm_per_pixel').value
        tops, seen_any = [], None
        for img, v in views:
            top, seen = topdown_from_view(img, v['K'], v['R'], v['t'], (x0, x1), (y0, y1), mm)
            tops.append(top)
            seen_any = seen if seen_any is None else (seen_any | seen)
        mosaic = np.minimum.reduce(tops)                       # darkest pixel = ink
        mosaic = cv2.flip(mosaic, 0)                           # far side of the table at the top
        path = os.path.join(folder, 'scan_topdown.png')
        cv2.imwrite(path, mosaic)
        coverage = 100.0 * seen_any.mean()
        self.get_logger().info(f'Top-down image saved: {path} '
                               f'({mosaic.shape[1]} x {mosaic.shape[0]} px, {mm} mm/px, '
                               f'{coverage:.0f}% of the area seen)')
        with open(os.path.join(folder, 'scan_area.yaml'), 'w') as f:
            yaml.safe_dump({'x_min': float(x0), 'x_max': float(x1), 'y_min': float(y0),
                            'y_max': float(y1), 'mm_per_pixel': float(mm)}, f)


def main():
    rclpy.init()
    node = ScanDrawing()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()