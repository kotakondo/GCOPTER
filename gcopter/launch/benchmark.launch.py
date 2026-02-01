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
        DeclareLaunchArgument("corridorMode", default_value="0"),
        DeclareLaunchArgument("corridorCacheFile", default_value=""),
        DeclareLaunchArgument("corridorBatchRoot", default_value=""),
        DeclareLaunchArgument("corridorBatchOutRoot", default_value=""),
        DeclareLaunchArgument("corridorBatchStart", default_value="0"),
        DeclareLaunchArgument("corridorBatchCount", default_value="0"),
        DeclareLaunchArgument("corridorBatchGoalWidth", default_value="3"),
        DeclareLaunchArgument("corridorNoMap", default_value="false"),
        DeclareLaunchArgument('start', default_value='[0.0, 0.0, 0.5]'),
        DeclareLaunchArgument('goal',  default_value='[10.0, 3.0, 1.5]'),
        DeclareLaunchArgument('max_vel',  default_value='2.0'),
        DeclareLaunchArgument("mighty_jerk_weight", default_value="1.0"),
        DeclareLaunchArgument("vel_ref_enable", default_value="false"),
        DeclareLaunchArgument("vel_ref_knot", default_value="-1"),
        DeclareLaunchArgument("vel_ref", default_value="[0.0, 0.0, 0.0]"),
        DeclareLaunchArgument("vel_ref_weight", default_value="0.0"),
        DeclareLaunchArgument("mighty_freeze_enable", default_value="false"),
        DeclareLaunchArgument("vel_ref_grad_check", default_value="false"),
        DeclareLaunchArgument("vel_ref_log_every", default_value="0"),
        DeclareLaunchArgument("pos_ref_enable", default_value="false"),
        DeclareLaunchArgument("pos_ref_knot", default_value="-1"),
        DeclareLaunchArgument("pos_ref", default_value="[0.0, 0.0, 0.0]"),
        DeclareLaunchArgument("pos_ref_weight", default_value="0.0"),
        DeclareLaunchArgument("pos_ref_grad_check", default_value="false"),
        DeclareLaunchArgument("pos_ref_log_every", default_value="0"),
        DeclareLaunchArgument("mighty_full_grad_check_enable", default_value="false"),
        DeclareLaunchArgument("mighty_full_grad_check_dirs", default_value="8"),
        DeclareLaunchArgument("mighty_full_grad_check_max_coords", default_value="256"),
        DeclareLaunchArgument("mighty_full_grad_check_eps", default_value="1e-5"),
        DeclareLaunchArgument("gcopter_full_grad_check_enable", default_value="false"),
        DeclareLaunchArgument("gcopter_full_grad_check_dirs", default_value="8"),
        DeclareLaunchArgument("gcopter_full_grad_check_max_coords", default_value="256"),
        DeclareLaunchArgument("gcopter_full_grad_check_eps", default_value="1e-5"),
        DeclareLaunchArgument("sample_dt", default_value="0.01"),
        DeclareLaunchArgument("collision_dt", default_value="0.01"),
        DeclareLaunchArgument("out_csv", default_value="/home/kkondo/data/gcopter_csv"),

        Node(
            package='gcopter',
            executable='minco_bench_viz',   # whatever you named the above file
            name='minco_bench_viz',
            output='screen',
            parameters=[gcopter_config_path,
                        {'corridorMode': LaunchConfiguration('corridorMode'),
                         'corridorCacheFile': LaunchConfiguration('corridorCacheFile'),
                         'corridorBatchRoot': LaunchConfiguration('corridorBatchRoot'),
                         'corridorBatchOutRoot': LaunchConfiguration('corridorBatchOutRoot'),
                         'corridorBatchStart': LaunchConfiguration('corridorBatchStart'),
                         'corridorBatchCount': LaunchConfiguration('corridorBatchCount'),
                         'corridorBatchGoalWidth': LaunchConfiguration('corridorBatchGoalWidth'),
                         'corridorNoMap': LaunchConfiguration('corridorNoMap'),
                         'Start': LaunchConfiguration('start'),
                         'Goal': LaunchConfiguration('goal'),
                         'MaxVelMag': LaunchConfiguration('max_vel'),
                         'MIGHTYJerkWeight': LaunchConfiguration('mighty_jerk_weight'),
                         'VelRefEnable': LaunchConfiguration('vel_ref_enable'),
                         'VelRefKnot': LaunchConfiguration('vel_ref_knot'),
                         'VelRef': LaunchConfiguration('vel_ref'),
                         'VelRefWeight': LaunchConfiguration('vel_ref_weight'),
                         'MIGHTYFreezeEnable': LaunchConfiguration('mighty_freeze_enable'),
                         'VelRefGradCheck': LaunchConfiguration('vel_ref_grad_check'),
                         'VelRefLogEvery': LaunchConfiguration('vel_ref_log_every'),
                         'PosRefEnable': LaunchConfiguration('pos_ref_enable'),
                         'PosRefKnot': LaunchConfiguration('pos_ref_knot'),
                         'PosRef': LaunchConfiguration('pos_ref'),
                         'PosRefWeight': LaunchConfiguration('pos_ref_weight'),
                         'PosRefGradCheck': LaunchConfiguration('pos_ref_grad_check'),
                         'PosRefLogEvery': LaunchConfiguration('pos_ref_log_every'),
                         'MIGHTYFullGradCheckEnable': LaunchConfiguration('mighty_full_grad_check_enable'),
                         'MIGHTYFullGradCheckDirs': LaunchConfiguration('mighty_full_grad_check_dirs'),
                         'MIGHTYFullGradCheckMaxCoords': LaunchConfiguration('mighty_full_grad_check_max_coords'),
                         'MIGHTYFullGradCheckEps': LaunchConfiguration('mighty_full_grad_check_eps'),
                         'GCOPTERFullGradCheckEnable': LaunchConfiguration('gcopter_full_grad_check_enable'),
                         'GCOPTERFullGradCheckDirs': LaunchConfiguration('gcopter_full_grad_check_dirs'),
                         'GCOPTERFullGradCheckMaxCoords': LaunchConfiguration('gcopter_full_grad_check_max_coords'),
                         'GCOPTERFullGradCheckEps': LaunchConfiguration('gcopter_full_grad_check_eps'),
                         'SampleDt': LaunchConfiguration('sample_dt'),
                         'CollisionDt': LaunchConfiguration('collision_dt'),
                         'ExportCSVDir': LaunchConfiguration('out_csv'),
                         'do_benchmark': True }],
            arguments=['--ros-args', '--log-level', 'minco_bench_viz:=debug']
        ),

    ])
