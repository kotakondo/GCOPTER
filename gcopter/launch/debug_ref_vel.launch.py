from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    gcopter_config_path = PathJoinSubstitution(
        [FindPackageShare('gcopter'), 'config', 'global_planning.yaml']
    )

    return LaunchDescription([
        DeclareLaunchArgument("corridor_cache", default_value=""),
        DeclareLaunchArgument("corridor_no_map", default_value="true"),
        DeclareLaunchArgument("do_benchmark", default_value="true"),
        DeclareLaunchArgument("max_vel", default_value="2.0"),
        DeclareLaunchArgument("mighty_jerk_weight", default_value="1e-3"),
        DeclareLaunchArgument("sample_dt", default_value="0.01"),
        DeclareLaunchArgument("collision_dt", default_value="0.01"),
        DeclareLaunchArgument("use_scaled_cost", default_value="true"),
        DeclareLaunchArgument("opt_timeout_ms", default_value="-1.0"),

        DeclareLaunchArgument("vel_ref_enable", default_value="true"),
        DeclareLaunchArgument("vel_ref_knot", default_value="1"),
        DeclareLaunchArgument("vel_ref", default_value="[0.0, 0.0, 0.0]"),
        DeclareLaunchArgument("vel_ref_weight", default_value="1.0"),
        DeclareLaunchArgument("mighty_freeze_enable", default_value="false"),
        DeclareLaunchArgument("vel_ref_grad_check", default_value="true"),
        DeclareLaunchArgument("vel_ref_log_every", default_value="0"),
        DeclareLaunchArgument("pos_ref_enable", default_value="false"),
        DeclareLaunchArgument("pos_ref_knot", default_value="1"),
        DeclareLaunchArgument("pos_ref_weight", default_value="1.0"),
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

        Node(
            package='gcopter',
            executable='minco_bench_viz',
            name='debug_ref_vel',
            output='screen',
            parameters=[gcopter_config_path,
                        {
                            'corridorMode': 2,
                            'corridorCacheFile': LaunchConfiguration('corridor_cache'),
                            'corridorNoMap': LaunchConfiguration('corridor_no_map'),
                            'do_benchmark': LaunchConfiguration('do_benchmark'),
                            'MaxVelMag': LaunchConfiguration('max_vel'),
                            'MIGHTYJerkWeight': LaunchConfiguration('mighty_jerk_weight'),
                            'SampleDt': LaunchConfiguration('sample_dt'),
                            'CollisionDt': LaunchConfiguration('collision_dt'),
                            'use_scaled_cost': LaunchConfiguration('use_scaled_cost'),
                            'opt_timeout_ms': LaunchConfiguration('opt_timeout_ms'),
                            'VelRefEnable': LaunchConfiguration('vel_ref_enable'),
                            'VelRefKnot': LaunchConfiguration('vel_ref_knot'),
                            'VelRef': LaunchConfiguration('vel_ref'),
                            'VelRefWeight': LaunchConfiguration('vel_ref_weight'),
                            'MIGHTYFreezeEnable': LaunchConfiguration('mighty_freeze_enable'),
                            'VelRefGradCheck': LaunchConfiguration('vel_ref_grad_check'),
                            'VelRefLogEvery': LaunchConfiguration('vel_ref_log_every'),
                            'PosRefEnable': LaunchConfiguration('pos_ref_enable'),
                            'PosRefKnot': LaunchConfiguration('pos_ref_knot'),
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
                        }],
            arguments=['--ros-args', '--log-level', 'debug_ref_vel:=info']
        ),
    ])
