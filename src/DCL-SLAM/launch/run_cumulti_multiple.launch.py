import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, TimerAction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    dcl_slam_share = get_package_share_directory('dcl_slam')

    # Ensure log output directory exists
    os.makedirs('/tmp/dcl_output', exist_ok=True)

    # FastDDS tuning for large PointCloud2 messages
    fastdds_profile = os.path.join(dcl_slam_share, 'config', 'fastdds_profile.xml')
    set_fastdds = SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', fastdds_profile)

    # Per-robot bag paths. Lidar uses the *_with_semantics bags produced by
    # extra_scripts/add_semantics_to_trimmed_bag.py (adds a uint16 `label`
    # PointField per scan). IMU/GPS bags are the original trimmed bags
    # (semantics pipeline does not touch them).
    bag_base = '/media/donceykong/donceys_data_ssd/datasets/CU_MULTI/main_campus/trimmed/'

    # Declare arguments
    bag_path_robot1_lidar_arg = DeclareLaunchArgument('bag_path_robot1_lidar',
        default_value=os.path.join(bag_base, 'robot1', 'robot1_main_campus_lidar_trimmed_with_semantics'),
        description='Path to robot1 trimmed lidar bag (with semantic label field)')
    bag_path_robot1_imu_gps_arg = DeclareLaunchArgument('bag_path_robot1_imu_gps',
        default_value=os.path.join(bag_base, 'robot1', 'robot1_main_campus_imu_gps_trimmed'),
        description='Path to robot1 trimmed imu_gps bag')
    bag_path_robot2_lidar_arg = DeclareLaunchArgument('bag_path_robot2_lidar',
        default_value=os.path.join(bag_base, 'robot2', 'robot2_main_campus_lidar_trimmed_with_semantics'),
        description='Path to robot2 trimmed lidar bag (with semantic label field)')
    bag_path_robot2_imu_gps_arg = DeclareLaunchArgument('bag_path_robot2_imu_gps',
        default_value=os.path.join(bag_base, 'robot2', 'robot2_main_campus_imu_gps_trimmed'),
        description='Path to robot2 trimmed imu_gps bag')
    bag_path_robot3_lidar_arg = DeclareLaunchArgument('bag_path_robot3_lidar',
        default_value=os.path.join(bag_base, 'robot3', 'robot3_main_campus_lidar_trimmed_with_semantics'),
        description='Path to robot3 trimmed lidar bag (with semantic label field)')
    bag_path_robot3_imu_gps_arg = DeclareLaunchArgument('bag_path_robot3_imu_gps',
        default_value=os.path.join(bag_base, 'robot3', 'robot3_main_campus_imu_gps_trimmed'),
        description='Path to robot3 trimmed imu_gps bag')
    bag_rate_arg = DeclareLaunchArgument('bag_rate', default_value='1.0',
        description='Bag playback rate')
    bag_delay_arg = DeclareLaunchArgument('bag_delay', default_value='8.0',
        description='Seconds to wait before starting bag playback')
    bag_start_arg = DeclareLaunchArgument('bag_start', default_value='0.0',
        description='Seconds to skip into the bag before playback')
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='true',
        description='Use simulated clock from bag playback')

    # Loop visualization node
    loop_vis_node = Node(
        package='dcl_slam',
        executable='dcl_slam_loopVisualizationNode',
        name='dcl_slam_loopVisualizationNode',
        output='screen',
        parameters=[{'number_of_robots': 3, 'use_sim_time': True}],
    )

    # RViz with the CU-MULTI display config
    rviz_config = os.path.join(dcl_slam_share, 'rviz', 'cu_multi.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen',
    )

    # Single UGV launch file path (CU-Multi variant)
    single_ugv_launch = os.path.join(dcl_slam_share, 'launch', 'single_ugv_cumulti.launch.py')

    # Robot A (robot1 -> a)
    robot_a = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(single_ugv_launch),
        launch_arguments={
            'robot_prefix': 'a',
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }.items(),
    )

    # Robot B (robot2 -> b)
    robot_b = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(single_ugv_launch),
        launch_arguments={
            'robot_prefix': 'b',
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }.items(),
    )

    # Robot C (robot3 -> c)
    robot_c = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(single_ugv_launch),
        launch_arguments={
            'robot_prefix': 'c',
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }.items(),
    )

    # Parallel per-robot bag playback (lidar = *_with_semantics, imu_gps = trimmed):
    #   Robot 1 lidar is the clock source (--clock 100)
    #   All other players do NOT publish clock
    #   Each robot has separate lidar and imu_gps bag players

    # Robot 1 lidar (clock source)
    bag_play_robot1_lidar = TimerAction(
        period=LaunchConfiguration('bag_delay'),
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'play',
                    LaunchConfiguration('bag_path_robot1_lidar'),
                    '--rate', LaunchConfiguration('bag_rate'),
                    '--clock', '100',
                    '--start-offset', LaunchConfiguration('bag_start'),
                    '--remap',
                    '/robot1/ouster/points:=/a/ouster/points',
                ],
                output='screen',
            ),
        ],
    )

    # Robot 1 imu_gps
    bag_play_robot1_imu_gps = TimerAction(
        period=LaunchConfiguration('bag_delay'),
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'play',
                    LaunchConfiguration('bag_path_robot1_imu_gps'),
                    '--rate', LaunchConfiguration('bag_rate'),
                    '--start-offset', LaunchConfiguration('bag_start'),
                    '--remap',
                    '/robot1/imu/data:=/a/imu/data',
                ],
                output='screen',
            ),
        ],
    )

    # Robot 2 lidar
    bag_play_robot2_lidar = TimerAction(
        period=LaunchConfiguration('bag_delay'),
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'play',
                    LaunchConfiguration('bag_path_robot2_lidar'),
                    '--rate', LaunchConfiguration('bag_rate'),
                    '--start-offset', LaunchConfiguration('bag_start'),
                    '--remap',
                    '/robot2/ouster/points:=/b/ouster/points',
                ],
                output='screen',
            ),
        ],
    )

    # Robot 2 imu_gps
    bag_play_robot2_imu_gps = TimerAction(
        period=LaunchConfiguration('bag_delay'),
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'play',
                    LaunchConfiguration('bag_path_robot2_imu_gps'),
                    '--rate', LaunchConfiguration('bag_rate'),
                    '--start-offset', LaunchConfiguration('bag_start'),
                    '--remap',
                    '/robot2/imu/data:=/b/imu/data',
                ],
                output='screen',
            ),
        ],
    )

    # Robot 3 lidar
    bag_play_robot3_lidar = TimerAction(
        period=LaunchConfiguration('bag_delay'),
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'play',
                    LaunchConfiguration('bag_path_robot3_lidar'),
                    '--rate', LaunchConfiguration('bag_rate'),
                    '--start-offset', LaunchConfiguration('bag_start'),
                    '--remap',
                    '/robot3/ouster/points:=/c/ouster/points',
                ],
                output='screen',
            ),
        ],
    )

    # Robot 3 imu_gps
    bag_play_robot3_imu_gps = TimerAction(
        period=LaunchConfiguration('bag_delay'),
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'play',
                    LaunchConfiguration('bag_path_robot3_imu_gps'),
                    '--rate', LaunchConfiguration('bag_rate'),
                    '--start-offset', LaunchConfiguration('bag_start'),
                    '--remap',
                    '/robot3/imu/data:=/c/imu/data',
                ],
                output='screen',
            ),
        ],
    )

    return LaunchDescription([
        set_fastdds,
        bag_path_robot1_lidar_arg,
        bag_path_robot1_imu_gps_arg,
        bag_path_robot2_lidar_arg,
        bag_path_robot2_imu_gps_arg,
        bag_path_robot3_lidar_arg,
        bag_path_robot3_imu_gps_arg,
        bag_rate_arg,
        bag_delay_arg,
        bag_start_arg,
        use_sim_time_arg,
        loop_vis_node,
        rviz_node,
        robot_a,
        robot_b,
        robot_c,
        bag_play_robot1_lidar,
        bag_play_robot1_imu_gps,
        bag_play_robot2_lidar,
        bag_play_robot2_imu_gps,
        bag_play_robot3_lidar,
        bag_play_robot3_imu_gps,
    ])
