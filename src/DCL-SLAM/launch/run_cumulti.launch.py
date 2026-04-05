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

    # Declare arguments
    bag_path_arg = DeclareLaunchArgument('bag_path',
        default_value='/media/donceykong/doncey_ssd_03/datasets/cu_multi/main_campus/main_campus_robots1_2_merged_bag',
        description='Path to CU-Multi ROS2 bag directory')
    bag_rate_arg = DeclareLaunchArgument('bag_rate', default_value='1.0',
        description='Bag playback rate')
    bag_delay_arg = DeclareLaunchArgument('bag_delay', default_value='8.0',
        description='Seconds to wait before starting bag playback')
    bag_start_arg = DeclareLaunchArgument('bag_start', default_value='500.0',
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

    # Bag playback with topic remapping:
    #   CU-Multi bag topics (robot1/robot2/robot3) -> DCL-SLAM namespaces (a/b/c)
    bag_play = TimerAction(
        period=LaunchConfiguration('bag_delay'),
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'play',
                    LaunchConfiguration('bag_path'),
                    '--rate', LaunchConfiguration('bag_rate'),
                    '--clock', '100',
                    '--read-ahead-queue-size', '10000',
                    '--start-offset', LaunchConfiguration('bag_start'),
                    '--remap',
                    '/robot1/ouster/points:=/a/ouster/points',
                    '/robot2/ouster/points:=/b/ouster/points',
                    '/robot3/ouster/points:=/c/ouster/points',
                    '/robot1/imu/data:=/a/imu/data',
                    '/robot2/imu/data:=/b/imu/data',
                    '/robot3/imu/data:=/c/imu/data',
                ],
                output='screen',
            ),
        ],
    )

    return LaunchDescription([
        set_fastdds,
        bag_path_arg,
        bag_rate_arg,
        bag_delay_arg,
        bag_start_arg,
        use_sim_time_arg,
        loop_vis_node,
        robot_a,
        robot_b,
        robot_c,
        bag_play,
    ])
