"""Move the UR5 so its wrist camera looks at the drawing area, then take a photo.

Run after draw_node has finished (simulation + UR5 MoveIt running):
  ros2 run pattern_drawer camera_view --ros-args -p use_sim_time:=true

Steps
  1. Read the drawing area ('center_square' by default) from ~/panda_workspace.yaml.
  2. Try camera positions between the UR5 and the drawing, higher and lower, and pick the
     first one MoveIt can reach with the camera pointing at the centre of the drawing.
     (Straight above the drawing is ~1.1 m from the UR5 base - beyond its 0.85 m reach,
     so the view is angled from the UR5's side.)
  3. Set the camera pyramid (camera_frustum 'distance') so it ends on the drawing.
  4. Save one camera image to ~/ur5_images/.
"""
import math
import os
import time

import rclpy
import yaml
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (Constraints, MotionPlanRequest, OrientationConstraint,
                             PositionConstraint)
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image
from shape_msgs.msg import SolidPrimitive

NS = '/ur5'
GROUP = 'ur5_ur_manipulator'
TOOL = 'ur5_tool0'
FRAME = 'world'
UR5_BASE = (-0.70, 0.0, 0.75)
TABLE_Z = 0.75
CAM_OFFSET = 0.01               # camera sits 1 cm in front of tool0 along tool0 z
VFOV = 2 * math.atan(math.tan(1.047 / 2) * 480 / 640)   # vertical field of view (rad)
SUCCESS = 1


def norm(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def look_at_quaternion(forward):
    """Orientation whose z axis points along 'forward' (tool0 z = camera viewing direction)."""
    z = norm(forward)
    x = norm(cross((0.0, 0.0, 1.0), z))      # horizontal, so the image is not rolled
    y = cross(z, x)
    m = [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]
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


class CameraView(Node):
    def __init__(self):
        super().__init__('camera_view')
        self.declare_parameter('workspace_file', os.path.expanduser('~/panda_workspace.yaml'))
        self.declare_parameter('area', 'center_square')
        self.declare_parameter('photo_dir', os.path.expanduser('~/ur5_images'))
        self.move = ActionClient(self, MoveGroup, f'{NS}/move_action')
        self.frustum_param = self.create_client(SetParameters, '/camera_frustum/set_parameters')
        self.last_image = None
        self.create_subscription(Image, '/ur5/camera/image_raw', self.on_image, 1)

    def on_image(self, msg):
        self.last_image = msg

    def wait(self, future, timeout=60.0):
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return future.result()

    def drawing_area(self):
        path = self.get_parameter('workspace_file').value
        key = self.get_parameter('area').value
        with open(path) as f:
            b = yaml.safe_load(f)[key]
        return b['x_min'], b['x_max'], b['y_min'], b['y_max']

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
        return res is not None and res.result.error_code.val == SUCCESS

    def set_frustum_distance(self, d):
        if not self.frustum_param.wait_for_service(timeout_sec=2.0):
            return
        req = SetParameters.Request()
        req.parameters = [Parameter(name='distance', value=ParameterValue(
            type=ParameterType.PARAMETER_DOUBLE, double_value=float(d)))]
        self.wait(self.frustum_param.call_async(req), timeout=5.0)

    def save_photo(self):
        self.last_image = None
        end = time.time() + 5.0
        while self.last_image is None and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.last_image is None:
            self.get_logger().warn('No camera image received - check /ur5/camera/image_raw')
            return
        try:
            import cv2
            from cv_bridge import CvBridge
        except ImportError:
            self.get_logger().warn('Install cv_bridge to save photos: '
                                   'sudo apt install ros-humble-cv-bridge python3-opencv')
            return
        img = CvBridge().imgmsg_to_cv2(self.last_image, desired_encoding='bgr8')
        folder = self.get_parameter('photo_dir').value
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, time.strftime('drawing_%Y%m%d_%H%M%S.png'))
        cv2.imwrite(path, img)
        self.get_logger().info(f'Photo saved: {path}')

    def run(self):
        x0, x1, y0, y1 = self.drawing_area()
        target = ((x0 + x1) / 2, (y0 + y1) / 2, TABLE_Z)
        side = max(x1 - x0, y1 - y0)
        self.get_logger().info(f'Drawing area centre ({target[0]:.2f}, {target[1]:.2f}), '
                               f'size {side:.2f} m')
        self.move.wait_for_server()

        # Candidate camera positions: part of the way from the drawing back toward the UR5,
        # at different heights. First reachable one wins.
        for back in (0.45, 0.55, 0.65, 0.75):
            for height in (0.50, 0.40, 0.60, 0.30):
                cam = (target[0] + (UR5_BASE[0] - target[0]) * back,
                       target[1] + (UR5_BASE[1] - target[1]) * back,
                       TABLE_Z + height)
                forward = norm(tuple(t - c for t, c in zip(target, cam)))
                tool = tuple(c - CAM_OFFSET * f for c, f in zip(cam, forward))
                dist = math.dist(cam, target)
                self.get_logger().info(f'Trying camera at ({cam[0]:.2f}, {cam[1]:.2f}, '
                                       f'{cam[2]:.2f}), {dist:.2f} m from the drawing')
                if self.move_tool(tool, look_at_quaternion(forward)):
                    seen = 2 * dist * math.tan(VFOV / 2)
                    self.get_logger().info(
                        f'Camera in place. View covers about {seen:.2f} m at the drawing centre '
                        f'(drawing is {side:.2f} m)' +
                        ('' if seen >= side else ' - the drawing is larger than the view'))
                    self.set_frustum_distance(dist)
                    time.sleep(1.0)              # let the image settle after the motion
                    self.save_photo()
                    return
        self.get_logger().error('No camera pose reachable - check the UR5 can plan at all '
                                '(Plan & Execute in the UR5 RViz)')


def main():
    rclpy.init()
    node = CameraView()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()