# launch/benchmark.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch.event_handlers.on_process_exit import OnProcessExit

def generate_launch_description():
    # Common config file
    gcopter_config_path = PathJoinSubstitution(
        [FindPackageShare('gcopter'), 'config', 'global_planning.yaml']
    )

    # Args
    args = [
        DeclareLaunchArgument('seed', default_value='1024'),
        DeclareLaunchArgument('start', default_value='[0.0, 0.0, 0.5]'),
        DeclareLaunchArgument('goal',  default_value='[10.0, 3.0, 1.5]'),
        DeclareLaunchArgument('max_vel', default_value='4.0'),
        DeclareLaunchArgument('mighty_jerk_weight', default_value='0.1'),
        DeclareLaunchArgument('sample_dt', default_value='0.02'),
        DeclareLaunchArgument('collision_dt', default_value='0.02'),
        DeclareLaunchArgument('out_csv', default_value='/tmp/bench/out.csv'),
        DeclareLaunchArgument('trial_id', default_value='0'),
    ]

    mock = Node(
        package='mockamap',
        executable='mockamap_node',
        name='mockamap_node',
        output='screen',
        parameters=[{
            'seed': LaunchConfiguration('seed'),
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
    )

    bench = Node(
        package='gcopter',
        executable='benchmark',
        name='benchmark',
        output='screen',
        parameters=[
            gcopter_config_path,
            {
                'Start': LaunchConfiguration('start'),
                'Goal':  LaunchConfiguration('goal'),
                'MaxVelMag': LaunchConfiguration('max_vel'),
                'MIGHTYJerkWeight': LaunchConfiguration('mighty_jerk_weight'),
                'SampleDt': LaunchConfiguration('sample_dt'),
                'CollisionDt': LaunchConfiguration('collision_dt'),
                'OutCSV': LaunchConfiguration('out_csv'),
                'QuitOnFinish': True,
                'TrialID': LaunchConfiguration('trial_id'),
                'MapSeed': LaunchConfiguration('seed'),
                'MapTopic': '/voxel_map',
            },
        ],
    )

    # Shutdown the launch system when the benchmark node exits
    handler = RegisterEventHandler(
        OnProcessExit(
            target_action=bench,
            on_exit=[Shutdown()]
        )
    )

    return LaunchDescription(args + [mock, bench, handler])
