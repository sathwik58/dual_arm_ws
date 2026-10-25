"""Show the UR5 camera's field of view as a pyramid (frustum), in RViz and in Gazebo.

  RViz   : Marker on /ur5/camera/frustum, attached to ur5_camera_optical_frame
  Gazebo : the same lines sent to Gazebo's /marker service, updated as the arm moves

The pyramid is computed from /ur5/camera/camera_info (pinhole model):
  a pixel (u, v) at distance d lies at  x = (u - cx) / fx * d,  y = (v - cy) / fy * d,  z = d
in the optical frame (z forward, x right, y down).
"""
import subprocess

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker

OPTICAL_FRAME = 'ur5_camera_optical_frame'


def rotate(q, v):
    """Rotate vector v = (x, y, z) by quaternion q (geometry_msgs Quaternion)."""
    qx, qy, qz, qw = q.x, q.y, q.z, q.w
    vx, vy, vz = v
    # t = 2 * cross(q.xyz, v)
    tx = 2 * (qy * vz - qz * vy)
    ty = 2 * (qz * vx - qx * vz)
    tz = 2 * (qx * vy - qy * vx)
    # v' = v + w * t + cross(q.xyz, t)
    return (vx + qw * tx + (qy * tz - qz * ty),
            vy + qw * ty + (qz * tx - qx * tz),
            vz + qw * tz + (qx * ty - qy * tx))


class CameraFrustum(Node):
    def __init__(self):
        super().__init__('camera_frustum')
        self.declare_parameter('distance', 0.6)        # pyramid length (m)
        self.declare_parameter('gazebo', True)          # also draw it in Gazebo
        self.declare_parameter('gazebo_rate', 4.0)      # Gazebo updates per second

        self.info = None
        self.create_subscription(CameraInfo, '/ur5/camera/camera_info', self.on_info, 10)
        self.pub = self.create_publisher(Marker, '/ur5/camera/frustum', 10)

        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.gz_proc = None

        self.create_timer(0.1, self.publish_rviz)                      # 10 Hz
        self.create_timer(1.0 / self.get_parameter('gazebo_rate').value, self.publish_gazebo)

    def on_info(self, msg):
        self.info = msg

    def lines(self):
        """Pairs of points (optical frame): 4 edges from the camera + 4 sides of the far end."""
        if self.info is None:
            return None
        k = self.info.k
        fx, fy, cx, cy = k[0], k[4], k[2], k[5]
        w, h = self.info.width, self.info.height
        d = self.get_parameter('distance').value
        c = [((u - cx) / fx * d, (v - cy) / fy * d, d) for u, v in ((0, 0), (w, 0), (w, h), (0, h))]
        o = (0.0, 0.0, 0.0)
        pairs = []
        for i in range(4):
            pairs += [o, c[i], c[i], c[(i + 1) % 4]]
        return pairs

    # ------------------------------------------------------------ RViz
    def publish_rviz(self):
        pairs = self.lines()
        if pairs is None:
            return
        m = Marker()
        m.header.frame_id = OPTICAL_FRAME
        m.ns, m.id = 'ur5_camera', 0
        m.type, m.action = Marker.LINE_LIST, Marker.ADD
        m.frame_locked = True
        m.pose.orientation.w = 1.0
        m.scale.x = 0.004
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 1.0, 0.2, 1.0
        m.points = [Point(x=p[0], y=p[1], z=p[2]) for p in pairs]
        self.pub.publish(m)

    # ------------------------------------------------------------ Gazebo
    def publish_gazebo(self):
        if not self.get_parameter('gazebo').value:
            return
        if self.gz_proc is not None and self.gz_proc.poll() is None:
            return                                   # previous update still running
        pairs = self.lines()
        if pairs is None:
            return
        try:
            tf = self.tf.lookup_transform('world', OPTICAL_FRAME, Time()).transform
        except Exception:
            return
        t = tf.translation
        world = []
        for p in pairs:                              # optical frame -> world frame
            r = rotate(tf.rotation, p)
            world.append((r[0] + t.x, r[1] + t.y, r[2] + t.z))

        pts = ', '.join(f'point: {{x: {x:.4f}, y: {y:.4f}, z: {z:.4f}}}' for x, y, z in world)
        req = ('ns: "ur5_camera_frustum", id: 1, action: ADD_MODIFY, type: LINE_LIST, '
               'material: {ambient: {r: 0.1, g: 1, b: 0.2, a: 1}, '
               'diffuse: {r: 0.1, g: 1, b: 0.2, a: 1}, emissive: {r: 0.1, g: 1, b: 0.2, a: 1}}, '
               + pts)
        self.gz_proc = subprocess.Popen(
            ['ign', 'service', '-s', '/marker',
             '--reqtype', 'ignition.msgs.Marker', '--reptype', 'ignition.msgs.Empty',
             '--timeout', '300', '--req', req],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    rclpy.init()
    rclpy.spin(CameraFrustum())


if __name__ == '__main__':
    main()