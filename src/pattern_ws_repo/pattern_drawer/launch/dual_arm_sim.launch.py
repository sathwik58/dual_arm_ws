import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def include(file):
    """Include one of this package's launch files."""
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(
        get_package_share_directory('pattern_drawer'), 'launch', file)))


def generate_launch_description():
    pkg = get_package_share_directory('pattern_drawer')

    # Panda meshes: Gazebo needs the folder that contains franka_description
    franka_path = os.path.dirname(get_package_share_directory('franka_description'))
    set_ign_path = SetEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', franka_path)
    set_gz_path = SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', franka_path)

    # Gazebo with the table world
    world = os.path.join(pkg, 'worlds', 'box.sdf')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': f'-r {world}'}.items(),
    )

    # Gazebo -> ROS bridges
    clock_bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock'],
        output='screen',
    )
    camera_bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        arguments=[
            '/ur5/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/ur5/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
        ],
        output='screen',
    )

    # Robots, then their MoveIt (delayed)
    ur5 = include('ur5_control.launch.py')
    panda = TimerAction(period=3.0, actions=[include('panda_control.launch.py')])
    ur5_moveit = TimerAction(period=8.0, actions=[include('ur5_move_group.launch.py')])
    panda_moveit = TimerAction(period=10.0, actions=[include('panda_move_group.launch.py')])

    # Helper nodes
    camera_frustum = Node(
        package='pattern_drawer', executable='camera_frustum',
        parameters=[{'use_sim_time': True, 'distance': 0.6}],
        output='screen',
    )
    pen_trail = Node(
        package='pattern_drawer', executable='pen_trail',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    table_scene = Node(
        package='pattern_drawer', executable='table_scene',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )


    return LaunchDescription([
        set_ign_path, set_gz_path,     # must come before gazebo
        gazebo,
        clock_bridge,
        camera_bridge,
        ur5,
        panda,
        ur5_moveit,
        panda_moveit,
        camera_frustum,
        pen_trail,
        table_scene
    ])