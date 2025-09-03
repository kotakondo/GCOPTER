# launch/global_planning.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def launch_setup(context, *args, **kwargs):
    # Resolve the launch arg value at runtime
    use_simple = LaunchConfiguration('use_simple_case_benchmark').perform(context).lower() in ('true', '1', 'yes')

    # Choose parameters
    if use_simple:
        x_length, y_length, map_type, fractal, road_width = 20, 20, 3, 3, 5.0
    else:
        x_length, y_length, map_type, fractal, road_width = 25, 25, 1, 1, 0.0

    rviz_config_path = PathJoinSubstitution(
        [FindPackageShare('gcopter'), 'config', 'global_planning.rviz']
    )

    # Build actions with concrete Python values
    return [
        Node(
            package='mockamap',
            executable='mockamap_node',
            name='mockamap_node',
            output='screen',
            parameters=[{
                'seed': 1024,
                'update_freq': 1.0,
                'resolution': 0.25,
                'x_length': x_length,
                'y_length': y_length,
                'z_length': 5,
                'type': map_type,
                'complexity': 0.025,
                'fill': 0.3,
                'fractal': fractal,
                'attenuation': 0.1,
                'road_width': road_width,
            }],
            remappings=[('/mock_map', '/voxel_map')],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz',
            output='screen',
            arguments=['-d', rviz_config_path],
        ),
    ]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_simple_case_benchmark',
            default_value='false',
            choices=['true', 'false'],
            description='Use smaller simple map with wider roads.'
        ),
        OpaqueFunction(function=launch_setup),
    ])
