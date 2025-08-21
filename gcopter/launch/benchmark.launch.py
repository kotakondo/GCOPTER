# launch/global_planning.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    
    gcopter_config_path = PathJoinSubstitution(
        [FindPackageShare('gcopter'), 'config', 'global_planning.yaml']
    )

    return LaunchDescription([
        DeclareLaunchArgument('start', default_value='[0.0, 0.0, 0.5]'),
        DeclareLaunchArgument('goal',  default_value='[10.0, 3.0, 1.5]'),
        DeclareLaunchArgument('max_vel',  default_value='2.0'),
        DeclareLaunchArgument("mighty_jerk_weight", default_value="1.0"),
        DeclareLaunchArgument("sample_dt", default_value="0.01"),
        DeclareLaunchArgument("collision_dt", default_value="0.01"),
        DeclareLaunchArgument("out_csv", default_value="/home/kkondo/data/gcopter_csv"),

        Node(
            package='gcopter',
            executable='minco_bench_viz',   # whatever you named the above file
            name='minco_bench_viz',
            output='screen',
            parameters=[gcopter_config_path,
                        {'Start': LaunchConfiguration('start'),
                         'Goal': LaunchConfiguration('goal'),
                         'MaxVelMag': LaunchConfiguration('max_vel'),
                         'MightyJerkWeight': LaunchConfiguration('mighty_jerk_weight'),
                         'SampleDt': LaunchConfiguration('sample_dt'),
                         'CollisionDt': LaunchConfiguration('collision_dt'),
                         'ExportCSVDir': LaunchConfiguration('out_csv'),
                         'do_benchmark': True }],
        ),

    ])