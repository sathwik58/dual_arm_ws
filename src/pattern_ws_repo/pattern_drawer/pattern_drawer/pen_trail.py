"""Ink: wherever the Panda's pen tip touches the table, leave a black line.

  - RViz : Marker (LINE_LIST) on /panda/pen_trail
  - Gazebo: thin black box models 'ink_N' on the table, so the UR5 camera can see them
  - Service /panda/clear_ink (std_srvs/Trigger): remove all ink from Gazebo and RViz

Started by dual_arm_sim.launch.py. Clear by hand:
  ros2 service call /panda/clear_ink std_srvs/srv/Trigger {}
"""
import math
import subprocess

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.time import Time
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker

WORLD = 'dual_arm_world'
TIP = 'panda_pen_tip'
TABLE_Z = 0.75              # table top
TOUCH = 0.003               # tip closer than 3 mm to the table = touching
STEP = 0.01                 # new ink piece every 1 cm of pen travel
DRAW_X = (-0.40, 0.55)      # only ink on the table area in front of the Panda
DRAW_Y = (-0.55, 0.55)


class PenTrail(Node):
    def __init__(self):
        super().__init__('pen_trail')
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.trail_pub = self.create_publisher(Marker, '/panda/pen_trail', 10)
        self.create_service(Trigger, '/panda/clear_ink', self.clear_ink)
        self.segments = []          # [(a, b), ...] for RViz
        self.last = None            # previous touching point (None = pen up)
        self.seg = 0                # counter for unique Gazebo names
        self.spawned = []           # names of ink models made by this node
        self.create_timer(0.03, self.update)
        self.create_timer(1.0, self.publish_trail)       # re-send for late RViz
        self.get_logger().info('pen_trail ready (clear with /panda/clear_ink)')

    # ------------------------------------------------------------ drawing
    def update(self):
        try:
            t = self.tf.lookup_transform('world', TIP, Time()).transform.translation
        except Exception:  # noqa: BLE001  (TF not ready yet)
            return
        touching = (t.z - TABLE_Z <= TOUCH and DRAW_X[0] <= t.x <= DRAW_X[1]
                    and DRAW_Y[0] <= t.y <= DRAW_Y[1])
        if not touching:
            self.last = None                              # pen lifted: next line is new
            return
        p = Point(x=t.x, y=t.y, z=TABLE_Z + 0.001)
        if self.last is None:
            self.last = p
            return
        if math.hypot(p.x - self.last.x, p.y - self.last.y) < STEP:
            return
        self.segments.append((self.last, p))
        self.gazebo_segment(self.last, p)
        self.last = p
        self.publish_trail()

    def publish_trail(self):
        if not self.segments:
            return
        m = Marker()
        m.header.frame_id = 'world'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id = 'pen_trail', 0
        m.type, m.action = Marker.LINE_LIST, Marker.ADD
        m.scale.x = 0.003
        m.color.a = 1.0                                    # black
        m.pose.orientation.w = 1.0
        for a, b in self.segments:
            m.points += [a, b]
        self.trail_pub.publish(m)

    def gazebo_segment(self, a, b):
        """Spawn a thin black box from a to b on the table (visible to cameras)."""
        length = math.hypot(b.x - a.x, b.y - a.y)
        if length < 1e-4:
            return
        self.seg += 1
        name = f'ink_{self.get_clock().now().nanoseconds}_{self.seg}'   # never repeats
        mx, my = (a.x + b.x) / 2, (a.y + b.y) / 2
        yaw = math.atan2(b.y - a.y, b.x - a.x)
        sdf = (f'<sdf version="1.8"><model name="{name}"><static>true</static>'
               f'<pose>{mx} {my} {TABLE_Z + 0.0005} 0 0 {yaw}</pose>'
               f'<link name="link"><visual name="visual"><cast_shadows>false</cast_shadows>'
               f'<geometry><box><size>{length + 0.003} 0.003 0.001</size></box></geometry>'
               f'<material><ambient>0 0 0 1</ambient><diffuse>0 0 0 1</diffuse></material>'
               f'</visual></link></model></sdf>')
        subprocess.Popen(['ign', 'service', '-s', f'/world/{WORLD}/create',
                          '--reqtype', 'ignition.msgs.EntityFactory',
                          '--reptype', 'ignition.msgs.Boolean', '--timeout', '1000',
                          '--req', f"sdf: '{sdf}'"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.spawned.append(name)

    # ------------------------------------------------------------ clearing
    def ink_models(self):
        """Every ink model in Gazebo (also ones from an earlier pen_trail run)."""
        names = set(self.spawned)
        try:
            out = subprocess.run(['ign', 'model', '--list'], capture_output=True,
                                 text=True, timeout=10).stdout
            for line in out.splitlines():
                n = line.strip().lstrip('-').strip()
                if n.startswith('ink_'):
                    names.add(n)
        except (OSError, subprocess.TimeoutExpired):
            pass
        return sorted(names)

    def clear_ink(self, request, response):
        names = self.ink_models()
        procs = []
        for name in names:
            procs.append(subprocess.Popen(
                ['ign', 'service', '-s', f'/world/{WORLD}/remove',
                 '--reqtype', 'ignition.msgs.Entity', '--reptype', 'ignition.msgs.Boolean',
                 '--timeout', '2000', '--req', f'name: "{name}" type: MODEL'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            if len(procs) >= 20:                          # 20 at a time
                for p in procs:
                    p.wait()
                procs = []
        for p in procs:
            p.wait()
        # old-style Gazebo GUI marker lines from the first pen_trail version
        subprocess.run(['ign', 'service', '-s', '/marker', '--reqtype', 'ignition.msgs.Marker',
                        '--reptype', 'ignition.msgs.Empty', '--timeout', '1000',
                        '--req', 'action: DELETE_ALL'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # RViz
        m = Marker()
        m.header.frame_id = 'world'
        m.action = Marker.DELETEALL
        self.trail_pub.publish(m)
        self.segments, self.spawned, self.last = [], [], None

        response.success = True
        response.message = f'Removed {len(names)} ink pieces'
        self.get_logger().info(response.message)
        return response


def main():
    rclpy.init()
    node = PenTrail()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()