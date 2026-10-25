"""Map where the Panda can draw on the table, and find the biggest rectangle and square.

Run (with the simulation and the Panda's MoveIt running):
  ros2 run pattern_drawer workspace_map --ros-args -p use_sim_time:=true

For every point of a grid over the table it asks MoveIt's IK (/panda/compute_ik):
"can the pen tip touch the table here, pointing straight down?"
  green squares = reachable, red squares = not reachable       (RViz: /panda/workspace)
Then it finds, using only green grid points:
  blue outline   = largest rectangle that is completely reachable
  yellow outline = largest square that is completely reachable
  cyan outline   = largest square centred in front of the Panda (on y = 0)
and saves both to a YAML file that draw_node can use.
"""
import math
import os

import rclpy
import yaml
from geometry_msgs.msg import Point, PoseStamped
from moveit_msgs.srv import GetPositionIK
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray

TABLE_Z = 0.75
TCP_Z = TABLE_Z + 0.005 + 0.10          # gap + pen length: pen tip touching the table
X_RANGE = (-0.95, 0.95)                 # whole table length (-1.0 .. 1.0)
Y_RANGE = (-0.55, 0.55)                 # whole table width  (-0.6 .. 0.6)
STEP = 0.05                             # 5 cm grid
YAWS = [0.0, math.pi / 2, math.pi, -math.pi / 2]
SUCCESS = 1


class WorkspaceMap(Node):
    def __init__(self):
        super().__init__('workspace_map')
        self.declare_parameter('output_file', os.path.expanduser('~/panda_workspace.yaml'))
        self.ik = self.create_client(GetPositionIK, '/panda/compute_ik')
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(MarkerArray, '/panda/workspace', latched)

    # ------------------------------------------------------------ IK test for one point
    def reachable(self, x, y):
        for yaw in YAWS:
            req = GetPositionIK.Request()
            r = req.ik_request
            r.group_name = 'panda_fer_arm'
            r.ik_link_name = 'panda_fer_hand_tcp'
            r.robot_state.is_diff = True            # start from the current state
            r.avoid_collisions = True
            r.timeout.nanosec = 50_000_000          # 50 ms per try
            p = PoseStamped()
            p.header.frame_id = 'world'
            p.pose.position.x, p.pose.position.y, p.pose.position.z = x, y, TCP_Z
            p.pose.orientation.x = math.cos(yaw / 2)   # pen down, turned by yaw
            p.pose.orientation.y = math.sin(yaw / 2)
            p.pose.orientation.z = 0.0
            p.pose.orientation.w = 0.0                 # default is 1.0, must be 0 here
            r.pose_stamped = p
            future = self.ik.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
            res = future.result()
            if res is not None and res.error_code.val == SUCCESS:
                return True
        return False

    # ------------------------------------------------------------ biggest shapes in the grid
    @staticmethod
    def largest_rectangle(grid):
        """Largest all-True rectangle. Returns (area, i0, i1, j0, j1), inclusive indices."""
        rows, cols = len(grid), len(grid[0])
        heights = [0] * cols
        best = (0, 0, -1, 0, -1)
        for i in range(rows):
            for j in range(cols):
                heights[j] = heights[j] + 1 if grid[i][j] else 0
            stack = []                                    # largest rectangle in histogram
            for j in range(cols + 1):
                h = heights[j] if j < cols else 0
                start = j
                while stack and stack[-1][1] >= h:
                    s, sh = stack.pop()
                    area = sh * (j - s)
                    if area > best[0]:
                        best = (area, i - sh + 1, i, s, j - 1)
                    start = s
                stack.append((start, h))
        return best

    @staticmethod
    def largest_square(grid):
        """Largest all-True square. Returns (size, i0, i1, j0, j1), inclusive indices."""
        rows, cols = len(grid), len(grid[0])
        dp = [[0] * cols for _ in range(rows)]
        best = (0, 0, -1, 0, -1)
        for i in range(rows):
            for j in range(cols):
                if grid[i][j]:
                    dp[i][j] = 1 if i == 0 or j == 0 else \
                        1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
                    if dp[i][j] > best[0]:
                        n = dp[i][j]
                        best = (n, i - n + 1, i, j - n + 1, j)
        return best

    @staticmethod
    def largest_centered_square(grid, cj):
        """Largest all-True square whose centre lies on column cj (y = 0, the Panda's front axis).
        Returns (half_cells, ci, cj): the square spans rows ci-k..ci+k and columns cj-k..cj+k."""
        rows, cols = len(grid), len(grid[0])
        best = (-1, 0, cj)
        for ci in range(rows):
            k = 0
            while (ci - k >= 0 and ci + k < rows and cj - k >= 0 and cj + k < cols and
                   all(grid[i][j] for i in range(ci - k, ci + k + 1)
                       for j in range(cj - k, cj + k + 1))):
                k += 1
            k -= 1                                        # last k that was fully reachable
            if k > best[0]:
                best = (k, ci, cj)
        return best

    # ------------------------------------------------------------ markers
    def squares(self, mid, pts, rgb):
        m = Marker()
        m.header.frame_id = 'world'
        m.ns, m.id = 'panda_workspace', mid
        m.type, m.action = Marker.CUBE_LIST, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = STEP * 0.9
        m.scale.z = 0.002
        m.color.r, m.color.g, m.color.b, m.color.a = (*rgb, 0.5)
        m.points = pts
        return m

    def outline(self, mid, box, rgb):
        x0, x1, y0, y1 = box
        z = TABLE_Z + 0.004
        m = Marker()
        m.header.frame_id = 'world'
        m.ns, m.id = 'panda_workspace', mid
        m.type, m.action = Marker.LINE_STRIP, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.006
        m.color.r, m.color.g, m.color.b, m.color.a = (*rgb, 1.0)
        m.points = [Point(x=x0, y=y0, z=z), Point(x=x1, y=y0, z=z), Point(x=x1, y=y1, z=z),
                    Point(x=x0, y=y1, z=z), Point(x=x0, y=y0, z=z)]
        return m

    # ------------------------------------------------------------ main
    def run(self):
        self.get_logger().info('Waiting for /panda/compute_ik ...')
        self.ik.wait_for_service()

        xs = [round(X_RANGE[0] + i * STEP, 3)
              for i in range(int(round((X_RANGE[1] - X_RANGE[0]) / STEP)) + 1)]
        ys = [round(Y_RANGE[0] + j * STEP, 3)
              for j in range(int(round((Y_RANGE[1] - Y_RANGE[0]) / STEP)) + 1)]
        self.get_logger().info(f'Testing {len(xs) * len(ys)} points, this takes a few minutes')

        grid, ok, bad = [], [], []
        for i, x in enumerate(xs):
            row = []
            for y in ys:
                r = self.reachable(x, y)
                row.append(r)
                (ok if r else bad).append(Point(x=x, y=y, z=TABLE_Z + 0.001))
            grid.append(row)
            self.get_logger().info(f'row {i + 1}/{len(xs)} (x = {x:.2f}) done')

        self.get_logger().info(f'{len(ok)} reachable, {len(bad)} unreachable points')
        markers = [self.squares(0, ok, (0.1, 0.9, 0.2)), self.squares(1, bad, (0.9, 0.1, 0.1))]
        result = {'grid_step': STEP, 'table_z': TABLE_Z}

        area, i0, i1, j0, j1 = self.largest_rectangle(grid)
        if area > 0:
            rect = (xs[i0], xs[i1], ys[j0], ys[j1])
            markers.append(self.outline(2, rect, (0.1, 0.4, 1.0)))
            result['rectangle'] = {'x_min': rect[0], 'x_max': rect[1],
                                   'y_min': rect[2], 'y_max': rect[3]}
            self.get_logger().info(
                f'Largest rectangle: x {rect[0]:.2f}..{rect[1]:.2f}, y {rect[2]:.2f}..{rect[3]:.2f}'
                f'  ({rect[1] - rect[0]:.2f} x {rect[3] - rect[2]:.2f} m)')

        n, i0, i1, j0, j1 = self.largest_square(grid)
        if n > 0:
            sq = (xs[i0], xs[i1], ys[j0], ys[j1])
            markers.append(self.outline(3, sq, (1.0, 0.85, 0.0)))
            result['square'] = {'x_min': sq[0], 'x_max': sq[1], 'y_min': sq[2], 'y_max': sq[3]}
            self.get_logger().info(
                f'Largest square:    x {sq[0]:.2f}..{sq[1]:.2f}, y {sq[2]:.2f}..{sq[3]:.2f}'
                f'  ({sq[1] - sq[0]:.2f} x {sq[3] - sq[2]:.2f} m)')

        # Largest square centred in front of the Panda (on the line y = 0)
        cj = min(range(len(ys)), key=lambda j: abs(ys[j]))
        k, ci, cj = self.largest_centered_square(grid, cj)
        if k >= 1:
            cs = (xs[ci - k], xs[ci + k], ys[cj - k], ys[cj + k])
            markers.append(self.outline(4, cs, (0.0, 0.9, 0.9)))
            result['center_square'] = {'x_min': cs[0], 'x_max': cs[1],
                                       'y_min': cs[2], 'y_max': cs[3]}
            self.get_logger().info(
                f'Centred square:    x {cs[0]:.2f}..{cs[1]:.2f}, y {cs[2]:.2f}..{cs[3]:.2f}'
                f'  ({cs[1] - cs[0]:.2f} x {cs[3] - cs[2]:.2f} m, centre x {xs[ci]:.2f}, y 0)')

        self.pub.publish(MarkerArray(markers=markers))

        path = self.get_parameter('output_file').value
        with open(path, 'w') as f:
            yaml.safe_dump(result, f, sort_keys=False)
        self.get_logger().info(f'Saved to {path}  (draw it with: draw_node -p shape:=center_square)')
        self.get_logger().info('Done. Markers stay visible in RViz; press Ctrl+C to quit.')


def main():
    rclpy.init()
    node = WorkspaceMap()
    node.run()
    rclpy.spin(node)        # keep the markers available for RViz


if __name__ == '__main__':
    main()