#include <rclcpp/rclcpp.hpp>
#include "dcl_slam/msg/loop_info.hpp"

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>

#include <thread>
#include <mutex>

using namespace std;

rclcpp::Node::SharedPtr node;
rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_loop_closure_constraints;
std::vector<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr> sub_keypose_cloud_vec;
std::vector<rclcpp::Subscription<dcl_slam::msg::LoopInfo>::SharedPtr> sub_loop_info_vec;
int number_of_robots_;
std::map<int, pcl::PointCloud<pcl::PointXYZI>> cloud_queue_vec;
std::map<gtsam::Symbol, gtsam::Symbol> loop_indexs;

void loopClosureThread()
{
	rclcpp::Rate rate(0.1);
	while(rclcpp::ok())
	{
		rate.sleep();

		if(loop_indexs.empty())
		{
			continue;
		}

		// loop nodes
		visualization_msgs::msg::Marker nodes;
		nodes.header.frame_id = "world";
		nodes.header.stamp = node->now();
		nodes.action = visualization_msgs::msg::Marker::ADD;
		nodes.type = visualization_msgs::msg::Marker::SPHERE_LIST;
		nodes.ns = "loop_nodes";
		nodes.id = 2;
		nodes.pose.orientation.w = 1;
		nodes.scale.x = 0.3; nodes.scale.y = 0.3; nodes.scale.z = 0.3;
		nodes.color.r = 0.93; nodes.color.g = 0.83; nodes.color.b = 0.0;
		nodes.color.a = 1;

		// loop edges
		visualization_msgs::msg::Marker constraints;
		constraints.header.frame_id = "world";
		constraints.header.stamp = node->now();
		constraints.action = visualization_msgs::msg::Marker::ADD;
		constraints.type = visualization_msgs::msg::Marker::LINE_LIST;
		constraints.ns = "loop_constraints";
		constraints.id = 3;
		constraints.pose.orientation.w = 1;
		constraints.scale.x = 0.1;
		constraints.color.r = 1.0; constraints.color.g = 0.91; constraints.color.b = 0.31;
		constraints.color.a = 1;

		pcl::PointXYZI pose_3d;
		int robot0, robot1, index0, index1;
		gtsam::Symbol key0, key1;
		for(auto it = loop_indexs.begin(); it != loop_indexs.end(); ++it)
        {
			key0 = it->first;
			key1 = it->second;
			robot0 = key0.chr() - 'a';
			robot1 = key1.chr() - 'a';
			index0 = key0.index();
			index1 = key1.index();

			if(index0 >= cloud_queue_vec[robot0].size() || index1 >= cloud_queue_vec[robot1].size())
			{
				continue;
			}

			geometry_msgs::msg::Point p;
			pose_3d = cloud_queue_vec[robot0].points[index0];
			p.x = pose_3d.x;
			p.y = pose_3d.y;
			p.z = pose_3d.z;
			nodes.points.push_back(p);
			constraints.points.push_back(p);
			pose_3d = cloud_queue_vec[robot1].points[index1];
			p.x = pose_3d.x;
			p.y = pose_3d.y;
			p.z = pose_3d.z;
			nodes.points.push_back(p);
			constraints.points.push_back(p);
		}

		// publish loop closure markers
		visualization_msgs::msg::MarkerArray markers_array;
		markers_array.markers.push_back(nodes);
		markers_array.markers.push_back(constraints);
		pub_loop_closure_constraints->publish(markers_array);
	}
}

void keyposeCloudHandler(
	const sensor_msgs::msg::PointCloud2::SharedPtr msg,
	int id)
{
	pcl::PointCloud<pcl::PointXYZI>::Ptr keyposes(new pcl::PointCloud<pcl::PointXYZI>());
	pcl::fromROSMsg(*msg, *keyposes);
	cloud_queue_vec[id] = *keyposes;
}

void loopInfoHandler(
	const dcl_slam::msg::LoopInfo::SharedPtr msg,
	int id)
{
	if((int)msg->noise != 999 && (int)msg->noise != 888)
	{
		gtsam::Symbol symbol0((msg->robot0 + 'a'), msg->index0);
		gtsam::Symbol symbol1((msg->robot1 + 'a'), msg->index1);
		auto it = loop_indexs.find(symbol0);
		if(it == loop_indexs.end() || (it != loop_indexs.end() && it->second != symbol1))
		{
			loop_indexs[symbol0] = symbol1;
		}
	}
}

int main(
	int argc,
	char** argv)
{
    rclcpp::init(argc, argv);
	node = rclcpp::Node::make_shared("loop_visualization_node");

	node->declare_parameter<int>("number_of_robots", 3);
	node->get_parameter("number_of_robots", number_of_robots_);

	for(int i = 0; i < number_of_robots_; i++)
	{
		std::string name = "a";
		name[0] += i;

		auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
		auto sub_keypose_cloud = node->create_subscription<sensor_msgs::msg::PointCloud2>(
			name+"/distributedMapping/keyposeCloud", qos,
			[i](const sensor_msgs::msg::PointCloud2::SharedPtr msg){ keyposeCloudHandler(msg, i); });
		sub_keypose_cloud_vec.push_back(sub_keypose_cloud);

		auto sub_loop_info = node->create_subscription<dcl_slam::msg::LoopInfo>(
			name+"/distributedMapping/loopInfo", qos,
			[i](const dcl_slam::msg::LoopInfo::SharedPtr msg){ loopInfoHandler(msg, i); });
		sub_loop_info_vec.push_back(sub_loop_info);
	}

	pub_loop_closure_constraints = node->create_publisher<visualization_msgs::msg::MarkerArray>(
		"distributedMapping/loopClosureConstraints",
		rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile());

	std::thread loop_visualization_thread(loopClosureThread);

	rclcpp::spin(node);

	loop_visualization_thread.join();

    return 0;
}
