# launch/global_planning.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    
    rviz_config_path = PathJoinSubstitution(
        [FindPackageShare('gcopter'), 'config', 'global_planning.rviz']
    )

    use_simple_case_benchmark = LaunchConfiguration('use_simple_case_benchmark', default='false')

    if use_simple_case_benchmark:
        x_length = 20
        y_length = 20
        type = 3
        fractal = 3
        road_width = 5.0
    else:
        x_length = 50
        y_length = 50
        type = 1
        fractal = 1
        road_width = 0.0

    return LaunchDescription([

        # mockamap (as you had it)
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
                'type': type,
                'complexity': 0.025,
                'fill': 0.3,
                'fractal': fractal,
                'attenuation': 0.1,
                'road_width': road_width,
            }],
            remappings=[('/mock_map', '/voxel_map')],
        ),

        # optional: rviz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz',
            output='screen',
            arguments=['-d', rviz_config_path],
        ),
    ])
