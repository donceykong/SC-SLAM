import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    dcl_slam_share = get_package_share_directory('dcl_slam')

    lio_type_arg = DeclareLaunchArgument('lio_type', default_value='1',
        description='1 for LIO-SAM, 2 for FAST-LIO2')

    # Loop visualization node
    loop_vis_node = Node(
        package='dcl_slam',
        executable='dcl_slam_loopVisualizationNode',
        name='dcl_slam_loopVisualizationNode',
        output='screen',
        parameters=[{'number_of_robots': 3}],
    )

    single_ugv_launch = os.path.join(dcl_slam_share, 'launch', 'single_ugv_kitti.launch.py')

    robot_a = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(single_ugv_launch),
        launch_arguments={'robot_prefix': 'a', 'lio_type': LaunchConfiguration('lio_type')}.items(),
    )

    robot_b = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(single_ugv_launch),
        launch_arguments={'robot_prefix': 'b', 'lio_type': LaunchConfiguration('lio_type')}.items(),
    )

    robot_c = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(single_ugv_launch),
        launch_arguments={'robot_prefix': 'c', 'lio_type': LaunchConfiguration('lio_type')}.items(),
    )

    # Bag playback: use ros2 bag play externally with --remap if needed
    # Example for KITTI sequence 05:
    #   ros2 bag play kitti_sequence_05_01 --remap topics...

    return LaunchDescription([
        lio_type_arg,
        loop_vis_node,
        robot_a,
        robot_b,
        robot_c,
    ])
