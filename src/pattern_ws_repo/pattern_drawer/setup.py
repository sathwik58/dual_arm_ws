from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'pattern_drawer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
            ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
            ('share/' + package_name, ['package.xml']),
            (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
            (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
            (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
            (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
            (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sathwik',
    maintainer_email='sathwikmunigala200@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'draw_node = pattern_drawer.draw_node:main',
            'camera_frustum = pattern_drawer.camera_frustum:main',
            'pen_trail = pattern_drawer.pen_trail:main',
            'table_scene = pattern_drawer.table_scene:main',
            'workspace_map = pattern_drawer.workspace_map:main',
            'camera_view = pattern_drawer.camera_view:main',
            'scan_drawing = pattern_drawer.scan_drawing:main',
            'train_shape_cnn = pattern_drawer.train_shape_cnn:main',
            'shape_recognizer = pattern_drawer.shape_reco:main',
            'run_report = pattern_drawer.run_report:main',
            'draw_gui = pattern_drawer.draw_gui:main',

        ],
    },
)
