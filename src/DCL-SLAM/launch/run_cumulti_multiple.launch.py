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

    # Per-robot bag paths
    bag_base = '/media/donceykong/doncey_ssd_03/datasets/cu_multi/main_campus'

    # Declare arguments
    bag_path_robot1_arg = DeclareLaunchArgument('bag_path_robot1',
        default_value=os.path.join(bag_base, 'robot1', 'robot1_main_campus_lidar_imu_gps'),
        description='Path to robot1 ROS2 bag')
    bag_path_robot2_arg = DeclareLaunchArgument('bag_path_robot2',
        default_value=os.path.join(bag_base, 'robot2', 'robot2_main_campus_lidar_imu_gps'),
        description='Path to robot2 ROS2 bag')
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

    # Parallel per-robot bag playback:
    #   Robot 1 is the clock source (--clock 100)
    #   Robot 2 does NOT publish clock (paces via --rate against wall time)
    #   Each player only reads its own robot's data from a separate bag file,
    #   eliminating the single-process I/O bottleneck.
    bag_play_robot1 = TimerAction(
        period=LaunchConfiguration('bag_delay'),
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'play',
                    LaunchConfiguration('bag_path_robot1'),
                    '--rate', LaunchConfiguration('bag_rate'),
                    '--clock', '100',
                    '--start-offset', LaunchConfiguration('bag_start'),
                    '--remap',
                    '/robot1/ouster/points:=/a/ouster/points',
                    '/robot1/imu/data:=/a/imu/data',
                ],
                output='screen',
            ),
        ],
    )

    bag_play_robot2 = TimerAction(
        period=LaunchConfiguration('bag_delay'),
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'play',
                    LaunchConfiguration('bag_path_robot2'),
                    '--rate', LaunchConfiguration('bag_rate'),
                    '--start-offset', LaunchConfiguration('bag_start'),
                    '--remap',
                    '/robot2/ouster/points:=/b/ouster/points',
                    '/robot2/imu/data:=/b/imu/data',
                ],
                output='screen',
            ),
        ],
    )

    return LaunchDescription([
        set_fastdds,
        bag_path_robot1_arg,
        bag_path_robot2_arg,
        bag_rate_arg,
        bag_delay_arg,
        bag_start_arg,
        use_sim_time_arg,
        loop_vis_node,
        robot_a,
        robot_b,
        robot_c,
        bag_play_robot1,
        bag_play_robot2,
    ])
