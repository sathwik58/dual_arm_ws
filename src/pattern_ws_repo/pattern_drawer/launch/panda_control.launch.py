import os
import re
import xml.dom.minidom
 
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
 
PANDA_XYZ = '0.70 0 0.75'
PANDA_RPY = '0 0 3.1416'
NAMESPACE = 'panda'
ARM_PREFIX = 'panda'          
 
def load_controller(context: LaunchContext):
    return [Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            'joint_trajectory_controller',
            '-c', f'/{NAMESPACE}/controller_manager',
            '--controller-manager-timeout', '30',
        ],
        output='screen',
    )]
def get_robot_description(context: LaunchContext, robot_type, load_gripper, franka_hand):
    robot_type_str = context.perform_substitution(robot_type)
    load_gripper_str = context.perform_substitution(load_gripper)
    franka_hand_str = context.perform_substitution(franka_hand)
 
    franka_xacro_file = os.path.join(
        get_package_share_directory('pattern_drawer'),
        'urdf', 'panda_pen.urdf.xacro'
    )
 
    robot_description_config = xacro.process_file(
        franka_xacro_file,
        mappings={
            'robot_type': robot_type_str,
            'hand': load_gripper_str,
            'gazebo': 'true',
            'ee_id': franka_hand_str,
            'gazebo_effort': 'false',      
            'arm_prefix': ARM_PREFIX,
            'xyz': PANDA_XYZ,
            'rpy': PANDA_RPY,
        }
    )
 
    if not isinstance(robot_description_config, xml.dom.minidom.Document):
        raise RuntimeError(
            f'The given xacro file {franka_xacro_file} is not a valid xml format.')
 
    urdf = robot_description_config.toxml()
    controllers = os.path.join(get_package_share_directory('pattern_drawer'),
                               'config', 'panda_controllers.yaml')
    urdf = re.sub(r'<parameters>[^<]*</parameters>',
                  f'<parameters>{controllers}</parameters>'
                  f'<ros><namespace>/{NAMESPACE}</namespace></ros>',
                  urdf)
    link4 = f'{ARM_PREFIX}_{robot_type_str}_link4'
    urdf = re.sub(rf'(<link name="{link4}">[\s\S]*?<inertia )[^>]*(/>)',
                  r'\1ixx="0.0078" ixy="0" ixz="0" iyy="0.0078" iyz="0" izz="0.0100"\2',
                  urdf)
 
    robot_description = {'robot_description': urdf}
 
    return [Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace=NAMESPACE,
        output='both',
        parameters=[robot_description, {'use_sim_time': True}],
    )]
 
 
def generate_launch_description():
    load_gripper_name = 'load_gripper'
    franka_hand_name = 'franka_hand'
    robot_type_name = 'robot_type'
 
    load_gripper = LaunchConfiguration(load_gripper_name)
    franka_hand = LaunchConfiguration(franka_hand_name)
    robot_type = LaunchConfiguration(robot_type_name)
 
    load_gripper_launch_argument = DeclareLaunchArgument(
        load_gripper_name,
        default_value='true',
        description='true/false for mounting the Franka hand')
    franka_hand_launch_argument = DeclareLaunchArgument(
        franka_hand_name,
        default_value='franka_hand',
        description='Default value: franka_hand')
    robot_type_launch_argument = DeclareLaunchArgument(
        robot_type_name,
        default_value='fer',
        description='fer = Franka Emika Panda (also: fr3, fp3)')
    robot_state_publisher = OpaqueFunction(
        function=get_robot_description,
        args=[robot_type, load_gripper, franka_hand])
 
    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', f'/{NAMESPACE}/robot_description', '-name', NAMESPACE],
        output='screen',
    )
 
    launch_controller = OpaqueFunction(function=load_controller)
 
    return LaunchDescription([
        load_gripper_launch_argument,
        franka_hand_launch_argument,
        robot_type_launch_argument,
        robot_state_publisher,
        spawn,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn,
                on_exit=[launch_controller],
            )
        ),
    ])