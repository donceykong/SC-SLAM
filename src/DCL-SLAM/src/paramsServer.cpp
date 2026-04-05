#include "paramsServer.h"

paramsServer::paramsServer(rclcpp::Node* node) : node_(node)
{
	// robot info
	std::string ns = node_->get_namespace(); // namespace of robot
	if(ns.length() >= 2 && ns[0] == '/' && ns[1] >= 'a' && ns[1] <= 'z')
	{
		name_ = ns.substr(1, 1); // remove '/' character
		id_ = name_[0]-'a'; // id of robot
	}
	else
	{
		// Fallback: try robot_id parameter
		node_->declare_parameter<std::string>("robot_id", "a");
		node_->get_parameter("robot_id", name_);
		id_ = name_[0]-'a';
		RCLCPP_WARN(node_->get_logger(), "Namespace '%s' not a valid robot prefix, using robot_id='%s' (id=%d)", ns.c_str(), name_.c_str(), id_);
	}

	node_->declare_parameter<int>("number_of_robots", 1);
	node_->get_parameter("number_of_robots", number_of_robots_);
	if(number_of_robots_ < 1)
	{
		RCLCPP_ERROR(node_->get_logger(), "Invalid robot number (must be positive number): %d", number_of_robots_);
		rclcpp::shutdown();
	}

	std::string param_prefix = "dcl_slam";
	// frames name
	node_->declare_parameter<std::string>(param_prefix + ".world_frame", "world");
	node_->declare_parameter<std::string>(param_prefix + ".odom_frame", "map");
	node_->get_parameter(param_prefix + ".world_frame", world_frame_);
	node_->get_parameter(param_prefix + ".odom_frame", odom_frame_);

	// lidar configuration
	std::string sensorStr;
	node_->declare_parameter<std::string>(param_prefix + ".sensor", "velodyne");
	node_->get_parameter(param_prefix + ".sensor", sensorStr);
	if(sensorStr == "velodyne")
	{
		sensor_ = LiDARType::VELODYNE;
	}
	else if(sensorStr == "ouster")
	{
		sensor_ = LiDARType::OUSTER;
	}
	else if(sensorStr == "livox")
	{
		sensor_ = LiDARType::LIVOX;
	}
	else
	{
		RCLCPP_ERROR(node_->get_logger(), "Invalid sensor type (must be either 'velodyne', 'ouster', or 'livox'): %s ", sensorStr.c_str());
		rclcpp::shutdown();
	}
	node_->declare_parameter<int>(param_prefix + ".n_scan", 16);
	node_->get_parameter(param_prefix + ".n_scan", n_scan_);

	// CPU Params
	node_->declare_parameter<int>(param_prefix + ".onboard_cpu_cores_num", 4);
	node_->declare_parameter<double>(param_prefix + ".loop_closure_process_interval", 0.02);
	node_->declare_parameter<double>(param_prefix + ".map_publish_interval", 10.0);
	node_->declare_parameter<double>(param_prefix + ".mapping_process_interval", 0.1);
	node_->get_parameter(param_prefix + ".onboard_cpu_cores_num", onboard_cpu_cores_num_);
	double tmp;
	node_->get_parameter(param_prefix + ".loop_closure_process_interval", tmp); loop_closure_process_interval_ = tmp;
	node_->get_parameter(param_prefix + ".map_publish_interval", tmp); map_publish_interval_ = tmp;
	node_->get_parameter(param_prefix + ".mapping_process_interval", tmp); mapping_process_interval_ = tmp;

	// mapping
	node_->declare_parameter<bool>(param_prefix + ".global_optmization_enable", false);
	node_->declare_parameter<bool>(param_prefix + ".use_pcm", false);
	node_->declare_parameter<double>(param_prefix + ".pcm_threshold", 0.75);
	node_->declare_parameter<bool>(param_prefix + ".use_between_noise", false);
	node_->declare_parameter<int>(param_prefix + ".optmization_maximum_iteration", 100);
	node_->declare_parameter<double>(param_prefix + ".failsafe_wait_time", 1.0);
	node_->declare_parameter<double>(param_prefix + ".rotation_estimate_change_threshold", 0.1);
	node_->declare_parameter<double>(param_prefix + ".pose_estimate_change_threshold", 0.1);
	node_->declare_parameter<double>(param_prefix + ".gamma", 1.0);
	node_->declare_parameter<bool>(param_prefix + ".use_flagged_init", true);
	node_->declare_parameter<bool>(param_prefix + ".use_landmarks", false);
	node_->declare_parameter<bool>(param_prefix + ".use_heuristics", true);

	node_->get_parameter(param_prefix + ".global_optmization_enable", global_optmization_enable_);
	node_->get_parameter(param_prefix + ".use_pcm", use_pcm_);
	node_->get_parameter(param_prefix + ".pcm_threshold", tmp); pcm_threshold_ = tmp;
	node_->get_parameter(param_prefix + ".use_between_noise", use_between_noise_);
	node_->get_parameter(param_prefix + ".optmization_maximum_iteration", optmization_maximum_iteration_);
	node_->get_parameter(param_prefix + ".failsafe_wait_time", tmp); fail_safe_wait_time_ = tmp;
	fail_safe_steps_ = fail_safe_wait_time_/mapping_process_interval_;
	node_->get_parameter(param_prefix + ".rotation_estimate_change_threshold", tmp); rotation_estimate_change_threshold_ = tmp;
	node_->get_parameter(param_prefix + ".pose_estimate_change_threshold", tmp); pose_estimate_change_threshold_ = tmp;
	node_->get_parameter(param_prefix + ".gamma", tmp); gamma_ = tmp;
	node_->get_parameter(param_prefix + ".use_flagged_init", use_flagged_init_);
	node_->get_parameter(param_prefix + ".use_landmarks", use_landmarks_);
	node_->get_parameter(param_prefix + ".use_heuristics", use_heuristics_);

	// downsample
	node_->declare_parameter<double>(param_prefix + ".map_leaf_size", 0.4);
	node_->declare_parameter<double>(param_prefix + ".descript_leaf_size", 0.1);
	node_->get_parameter(param_prefix + ".map_leaf_size", tmp); map_leaf_size_ = tmp;
	node_->get_parameter(param_prefix + ".descript_leaf_size", tmp); descript_leaf_size_ = tmp;

	// loop closure
	node_->declare_parameter<bool>(param_prefix + ".intra_robot_loop_closure_enable", true);
	node_->declare_parameter<bool>(param_prefix + ".inter_robot_loop_closure_enable", true);
	node_->declare_parameter<std::string>(param_prefix + ".descriptor_type", "");
	node_->declare_parameter<int>(param_prefix + ".knn_candidates", 10);
	node_->declare_parameter<int>(param_prefix + ".exclude_recent_frame_num", 30);
	node_->declare_parameter<double>(param_prefix + ".search_radius", 15.0);
	node_->declare_parameter<int>(param_prefix + ".match_mode", 2);
	node_->declare_parameter<int>(param_prefix + ".iris_row", 80);
	node_->declare_parameter<int>(param_prefix + ".iris_column", 360);
	node_->declare_parameter<double>(param_prefix + ".descriptor_distance_threshold", 0.4);
	node_->declare_parameter<int>(param_prefix + ".history_keyframe_search_num", 16);
	node_->declare_parameter<double>(param_prefix + ".fitness_score_threshold", 0.2);
	node_->declare_parameter<int>(param_prefix + ".ransac_maximum_iteration", 1000);
	node_->declare_parameter<double>(param_prefix + ".ransac_threshold", 0.5);
	node_->declare_parameter<double>(param_prefix + ".ransac_outlier_reject_threshold", 0.05);

	node_->get_parameter(param_prefix + ".intra_robot_loop_closure_enable", intra_robot_loop_closure_enable_);
	node_->get_parameter(param_prefix + ".inter_robot_loop_closure_enable", inter_robot_loop_closure_enable_);
	std::string descriptor_type_;
	node_->get_parameter(param_prefix + ".descriptor_type", descriptor_type_);
	if(descriptor_type_ == "ScanContext")
	{
		descriptor_type_num_ = DescriptorType::ScanContext;
	}
	else if(descriptor_type_ == "LidarIris")
	{
		descriptor_type_num_ = DescriptorType::LidarIris;
	}
	else if(descriptor_type_ == "M2DP")
	{
		descriptor_type_num_ = DescriptorType::M2DP;
	}
	else
	{
		inter_robot_loop_closure_enable_ = false;
		RCLCPP_WARN(node_->get_logger(), "Invalid descriptor type: %s, turn off interloop...", descriptor_type_.c_str());
	}
	node_->get_parameter(param_prefix + ".knn_candidates", knn_candidates_);
	node_->get_parameter(param_prefix + ".exclude_recent_frame_num", exclude_recent_frame_num_);
	node_->get_parameter(param_prefix + ".search_radius", tmp); search_radius_ = tmp;
	node_->get_parameter(param_prefix + ".match_mode", match_mode_);
	node_->get_parameter(param_prefix + ".iris_row", iris_row_);
	node_->get_parameter(param_prefix + ".iris_column", iris_column_);
	node_->get_parameter(param_prefix + ".descriptor_distance_threshold", tmp); descriptor_distance_threshold_ = tmp;
	node_->get_parameter(param_prefix + ".history_keyframe_search_num", history_keyframe_search_num_);
	node_->get_parameter(param_prefix + ".fitness_score_threshold", tmp); fitness_score_threshold_ = tmp;
	node_->get_parameter(param_prefix + ".ransac_maximum_iteration", ransac_maximum_iteration_);
	node_->get_parameter(param_prefix + ".ransac_threshold", tmp); ransac_threshold_ = tmp;
	node_->get_parameter(param_prefix + ".ransac_outlier_reject_threshold", tmp); ransac_outlier_reject_threshold_ = tmp;

	// keyframe params
	node_->declare_parameter<double>(param_prefix + ".keyframe_distance_threshold", 1.0);
	node_->declare_parameter<double>(param_prefix + ".keyframe_angle_threshold", 0.2);
	node_->get_parameter(param_prefix + ".keyframe_distance_threshold", tmp); keyframe_distance_threshold_ = tmp;
	node_->get_parameter(param_prefix + ".keyframe_angle_threshold", tmp); keyframe_angle_threshold_ = tmp;

	// visualization
	node_->declare_parameter<double>(param_prefix + ".global_map_visualization_radius", 60.0);
	node_->get_parameter(param_prefix + ".global_map_visualization_radius", tmp); global_map_visualization_radius_ = tmp;

	// output directory
	node_->declare_parameter<std::string>(param_prefix + ".save_directory", "/tmp/dcl_output");
	node_->get_parameter(param_prefix + ".save_directory", save_directory_);
}


Eigen::Affine3f paramsServer::gtsamPoseToAffine3f(gtsam::Pose3 pose)
{
	return pcl::getTransformation(pose.translation().x(), pose.translation().y(), pose.translation().z(),
		pose.rotation().roll(), pose.rotation().pitch(), pose.rotation().yaw());
}

geometry_msgs::msg::Transform paramsServer::gtsamPoseToTransform(gtsam::Pose3 pose)
{
	geometry_msgs::msg::Transform transform_msg;
	transform_msg.translation.x = pose.translation().x();
	transform_msg.translation.y = pose.translation().y();
	transform_msg.translation.z = pose.translation().z();
	transform_msg.rotation.w = pose.rotation().toQuaternion().w();
	transform_msg.rotation.x = pose.rotation().toQuaternion().x();
	transform_msg.rotation.y = pose.rotation().toQuaternion().y();
	transform_msg.rotation.z = pose.rotation().toQuaternion().z();

	return transform_msg;
}


gtsam::Pose3 paramsServer::transformToGtsamPose(const geometry_msgs::msg::Transform& pose)
{
	return gtsam::Pose3(gtsam::Rot3::Quaternion(pose.rotation.w, pose.rotation.x, pose.rotation.y, pose.rotation.z),
		gtsam::Point3(pose.translation.x, pose.translation.y, pose.translation.z));
}


gtsam::Pose3 paramsServer::pclPointTogtsamPose3(PointPose6D point)
{
	return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(point.roll), double(point.pitch), double(point.yaw)),
		gtsam::Point3(double(point.x), double(point.y), double(point.z)));
}


pcl::PointCloud<PointPose3D>::Ptr paramsServer::transformPointCloud(pcl::PointCloud<PointPose3D> cloud_in, PointPose6D* pose)
{
	pcl::PointCloud<PointPose3D>::Ptr cloud_out(new pcl::PointCloud<PointPose3D>());

	int cloud_size = cloud_in.size();
	cloud_out->resize(cloud_size);

	Eigen::Affine3f trans_cur = pcl::getTransformation(pose->x, pose->y, pose->z, pose->roll, pose->pitch, pose->yaw);

	#pragma omp parallel for num_threads(onboard_cpu_cores_num_)
	for(int i = 0; i < cloud_size; ++i)
	{
		const auto &p_from = cloud_in.points[i];
		cloud_out->points[i].x = trans_cur(0,0)*p_from.x + trans_cur(0,1)*p_from.y + trans_cur(0,2)*p_from.z + trans_cur(0,3);
		cloud_out->points[i].y = trans_cur(1,0)*p_from.x + trans_cur(1,1)*p_from.y + trans_cur(1,2)*p_from.z + trans_cur(1,3);
		cloud_out->points[i].z = trans_cur(2,0)*p_from.x + trans_cur(2,1)*p_from.y + trans_cur(2,2)*p_from.z + trans_cur(2,3);
		cloud_out->points[i].intensity = p_from.intensity;
	}
	return cloud_out;
}

pcl::PointCloud<PointPose3D>::Ptr paramsServer::transformPointCloud(pcl::PointCloud<PointPose3D> cloud_in, gtsam::Pose3 pose)
{
	pcl::PointCloud<PointPose3D>::Ptr cloud_out(new pcl::PointCloud<PointPose3D>());

	int cloud_size = cloud_in.size();
	cloud_out->resize(cloud_size);

	Eigen::Affine3f trans_cur = pcl::getTransformation(pose.translation().x(), pose.translation().y(), pose.translation().z(),
		pose.rotation().roll(), pose.rotation().pitch(), pose.rotation().yaw());

	#pragma omp parallel for num_threads(onboard_cpu_cores_num_)
	for(int i = 0; i < cloud_size; ++i)
	{
		const auto &p_from = cloud_in.points[i];
		cloud_out->points[i].x = trans_cur(0,0)*p_from.x + trans_cur(0,1)*p_from.y + trans_cur(0,2)*p_from.z + trans_cur(0,3);
		cloud_out->points[i].y = trans_cur(1,0)*p_from.x + trans_cur(1,1)*p_from.y + trans_cur(1,2)*p_from.z + trans_cur(1,3);
		cloud_out->points[i].z = trans_cur(2,0)*p_from.x + trans_cur(2,1)*p_from.y + trans_cur(2,2)*p_from.z + trans_cur(2,3);
		cloud_out->points[i].intensity = p_from.intensity;
	}
	return cloud_out;
}
