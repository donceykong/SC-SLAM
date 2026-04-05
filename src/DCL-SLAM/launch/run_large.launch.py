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
        parameters=[{'number_of_robots': 9}],
    )

    single_ugv_launch = os.path.join(dcl_slam_share, 'launch', 'single_ugv.launch.py')

    # 9 robots: a through i
    robot_prefixes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    robot_launches = []
    for prefix in robot_prefixes:
        robot_launches.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(single_ugv_launch),
                launch_arguments={
                    'robot_prefix': prefix,
                    'lio_type': LaunchConfiguration('lio_type'),
                }.items(),
            )
        )

    # Bag playback: use ros2 bag play externally with --remap if needed

    return LaunchDescription([
        lio_type_arg,
        loop_vis_node,
    ] + robot_launches)
