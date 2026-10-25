import rclpy
from rclpy.node import Node
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

TABLE_SIZE = (2.0, 1.2)     # x, y (m), same as box.sdf
TABLE_TOP = 0.745           # 5 mm below the real top (0.75) so the pen may touch
NAMESPACES = ['ur5', 'panda']


class TableScene(Node):
    def __init__(self):
        super().__init__('table_scene')
        self.pubs = [self.create_publisher(CollisionObject, f'/{ns}/collision_object', 10)
                     for ns in NAMESPACES]
        self.create_timer(2.0, self.publish)    # repeat, so a MoveIt that starts later still gets it

    def publish(self):
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [TABLE_SIZE[0], TABLE_SIZE[1], TABLE_TOP]

        pose = Pose()
        pose.position.z = TABLE_TOP / 2       # box centre: halfway up
        pose.orientation.w = 1.0

        obj = CollisionObject()
        obj.header.frame_id = 'world'
        obj.id = 'table'
        obj.primitives = [box]
        obj.primitive_poses = [pose]
        obj.operation = CollisionObject.ADD

        for pub in self.pubs:
            pub.publish(obj)


def main():
    rclpy.init()
    rclpy.spin(TableScene())