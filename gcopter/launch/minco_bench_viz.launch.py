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
    
    gcopter_config_path = PathJoinSubstitution(
        [FindPackageShare('gcopter'), 'config', 'global_planning.yaml']
    )

    return LaunchDescription([
        DeclareLaunchArgument('start', default_value='[0.0, 0.0, 0.5]'),
        DeclareLaunchArgument('goal',  default_value='[10.0, 3.0, 1.5]'),

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
                'x_length': 50,
                'y_length': 50,
                'z_length': 5,
                'type': 1,
                'complexity': 0.025,
                'fill': 0.3,
                'fractal': 1,
                'attenuation': 0.1,
            }],
            remappings=[('/mock_map', '/voxel_map')],
        ),

        Node(
            package='gcopter',
            executable='minco_bench_viz',   # whatever you named the above file
            name='minco_bench_viz',
            output='screen',
            parameters=[gcopter_config_path,
                        {'Start': LaunchConfiguration('start'),
                         'Goal': LaunchConfiguration('goal')}],
            prefix='xterm -e gdb -q -ex run --args', # gdb debugging
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
