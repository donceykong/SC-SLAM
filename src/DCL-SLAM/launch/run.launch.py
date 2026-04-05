import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, TimerAction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    dcl_slam_share = get_package_share_directory('dcl_slam')

    # FastDDS tuning for large PointCloud2 messages
    fastdds_profile = os.path.join(dcl_slam_share, 'config', 'fastdds_profile.xml')
    set_fastdds = SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', fastdds_profile)

    # Declare arguments
    lio_type_arg = DeclareLaunchArgument('lio_type', default_value='1',
        description='1 for LIO-SAM, 2 for FAST-LIO2')
    bag_path_arg = DeclareLaunchArgument('bag_path',
        default_value=os.path.expanduser('~/rosbag-data/S3E/S3E_Library_1'),
        description='Path to ROS2 bag directory')
    bag_rate_arg = DeclareLaunchArgument('bag_rate', default_value='1.0',
        description='Bag playback rate')
    bag_delay_arg = DeclareLaunchArgument('bag_delay', default_value='8.0',
        description='Seconds to wait before starting bag playback')
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

    # Single UGV launch file path
    single_ugv_launch = os.path.join(dcl_slam_share, 'launch', 'single_ugv.launch.py')

    # Robot A
    robot_a = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(single_ugv_launch),
        launch_arguments={
            'robot_prefix': 'a',
            'lio_type': LaunchConfiguration('lio_type'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }.items(),
    )

    # Robot B
    robot_b = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(single_ugv_launch),
        launch_arguments={
            'robot_prefix': 'b',
            'lio_type': LaunchConfiguration('lio_type'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }.items(),
    )

    # Robot C
    robot_c = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(single_ugv_launch),
        launch_arguments={
            'robot_prefix': 'c',
            'lio_type': LaunchConfiguration('lio_type'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }.items(),
    )

    # Bag playback with topic remapping and /clock publishing
    bag_play = TimerAction(
        period=LaunchConfiguration('bag_delay'),
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'play',
                    LaunchConfiguration('bag_path'),
                    '--rate', LaunchConfiguration('bag_rate'),
                    '--clock', '100',
                    '--remap',
                    '/Alpha/velodyne_points:=/a/velodyne_points',
                    '/Bob/velodyne_points:=/b/velodyne_points',
                    '/Carol/velodyne_points:=/c/velodyne_points',
                    '/Alpha/imu/data:=/a/imu/data',
                    '/Bob/imu/data:=/b/imu/data',
                    '/Carol/imu/data:=/c/imu/data',
                ],
                output='screen',
            ),
        ],
    )

    return LaunchDescription([
        set_fastdds,
        lio_type_arg,
        bag_path_arg,
        bag_rate_arg,
        bag_delay_arg,
        use_sim_time_arg,
        loop_vis_node,
        robot_a,
        robot_b,
        robot_c,
        bag_play,
    ])
