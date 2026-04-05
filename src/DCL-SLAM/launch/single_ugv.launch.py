import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory
import xacro


def generate_launch_description():
    # Declare arguments
    lio_type_arg = DeclareLaunchArgument('lio_type', default_value='1',
        description='1 for LIO-SAM, 2 for FAST-LIO2')
    robot_prefix_arg = DeclareLaunchArgument('robot_prefix', default_value='a')
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='true')

    return LaunchDescription([
        lio_type_arg,
        robot_prefix_arg,
        use_sim_time_arg,
        OpaqueFunction(function=launch_setup),
    ])


def launch_setup(context):
    lio_type = int(LaunchConfiguration('lio_type').perform(context))
    robot_prefix = LaunchConfiguration('robot_prefix').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'

    dcl_slam_share = get_package_share_directory('dcl_slam')
    sim_time_param = {'use_sim_time': use_sim_time}

    actions = []

    if lio_type == 1:
        # LIO-SAM configuration
        params_file = os.path.join(dcl_slam_share, 'config',
            f'dcl_lio_sam_vlp16_params_{robot_prefix}.yaml')

        # URDF for robot_state_publisher
        xacro_file = os.path.join(dcl_slam_share, 'config', 'lio_sam_robot.urdf.xacro')
        robot_description = ''
        if os.path.exists(xacro_file):
            robot_description = xacro.process_file(xacro_file).toxml()

        lio_sam_nodes = GroupAction([
            PushRosNamespace(robot_prefix),
            Node(
                package='dcl_lio_sam',
                executable='dcl_lio_sam_imuPreintegration',
                name='dcl_lio_sam_imuPreintegration',
                output='screen',
                parameters=[params_file, sim_time_param],
            ),
            Node(
                package='dcl_lio_sam',
                executable='dcl_lio_sam_imageProjection',
                name='dcl_lio_sam_imageProjection',
                output='screen',
                parameters=[params_file, sim_time_param],
            ),
            Node(
                package='dcl_lio_sam',
                executable='dcl_lio_sam_featureExtraction',
                name='dcl_lio_sam_featureExtraction',
                output='screen',
                parameters=[params_file, sim_time_param],
            ),
            Node(
                package='dcl_lio_sam',
                executable='dcl_lio_sam_mapOptmization',
                name='dcl_lio_sam_mapOptmization',
                output='screen',
                parameters=[params_file, sim_time_param],
            ),
            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                name='robot_state_publisher',
                parameters=[{'robot_description': robot_description}, sim_time_param],
            ),
        ])
        actions.append(lio_sam_nodes)

    elif lio_type == 2:
        # FAST-LIO2 configuration
        params_file = os.path.join(dcl_slam_share, 'config', 'dcl_fast_lio_velodyne16.yaml')

        fast_lio_params = {
            'feature_extract_enable': False,
            'point_filter_num': 1,
            'max_iteration': 3,
            'filter_size_corner': 0.2,
            'filter_size_surf': 0.2,
            'filter_size_map': 0.2,
            'cube_side_length': 1000.0,
            'runtime_pos_log_enable': False,
        }

        fast_lio_nodes = GroupAction([
            PushRosNamespace(robot_prefix),
            Node(
                package='dcl_fast_lio',
                executable='fastlio_mapping',
                name='laserMapping',
                output='screen',
                parameters=[params_file, fast_lio_params, sim_time_param],
            ),
        ])
        actions.append(fast_lio_nodes)

    return actions
