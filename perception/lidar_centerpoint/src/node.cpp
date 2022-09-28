// Copyright 2021 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lidar_centerpoint/node.hpp"

#include "perception_utils/object_classification.hpp"
#include "perception_utils/perception_utils.hpp"

#include <lidar_centerpoint/centerpoint_config.hpp>
#include <lidar_centerpoint/preprocess/pointcloud_densification.hpp>
#include <lidar_centerpoint/ros_utils.hpp>
#include <lidar_centerpoint/utils.hpp>
#include <pcl_ros/transforms.hpp>

#include <pcl/filters/random_sample.h>
#include <pcl/filters/voxel_grid.h>

#ifdef ROS_DISTRO_GALACTIC
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#else
#include <tf2_eigen/tf2_eigen.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#endif

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <memory>
#include <string>
#include <vector>

namespace centerpoint
{
LidarCenterPointNode::LidarCenterPointNode(const rclcpp::NodeOptions & node_options)
: Node("lidar_center_point", node_options), tf_buffer_(this->get_clock())
{
  const float score_threshold =
    static_cast<float>(this->declare_parameter<double>("score_threshold", 0.35));
  const float circle_nms_dist_threshold =
    static_cast<float>(this->declare_parameter<double>("circle_nms_dist_threshold"));
  const auto yaw_norm_thresholds =
    this->declare_parameter<std::vector<double>>("yaw_norm_thresholds");
  const std::string densification_world_frame_id =
    this->declare_parameter("densification_world_frame_id", "map");
  const int densification_num_past_frames =
    this->declare_parameter("densification_num_past_frames", 1);
  const std::string trt_precision = this->declare_parameter("trt_precision", "fp16");
  const std::string encoder_onnx_path = this->declare_parameter<std::string>("encoder_onnx_path");
  const std::string encoder_engine_path =
    this->declare_parameter<std::string>("encoder_engine_path");
  const std::string head_onnx_path = this->declare_parameter<std::string>("head_onnx_path");
  const std::string head_engine_path = this->declare_parameter<std::string>("head_engine_path");
  class_names_ = this->declare_parameter<std::vector<std::string>>("class_names");
  has_twist_ = this->declare_parameter("has_twist", false);
  const std::size_t point_feature_size =
    static_cast<std::size_t>(this->declare_parameter<std::int64_t>("point_feature_size"));
  const std::size_t max_voxel_size =
    static_cast<std::size_t>(this->declare_parameter<std::int64_t>("max_voxel_size"));
  const auto point_cloud_range = this->declare_parameter<std::vector<double>>("point_cloud_range");
  const auto voxel_size = this->declare_parameter<std::vector<double>>("voxel_size");
  const std::size_t downsample_factor =
    static_cast<std::size_t>(this->declare_parameter<std::int64_t>("downsample_factor"));
  const std::size_t encoder_in_feature_size =
    static_cast<std::size_t>(this->declare_parameter<std::int64_t>("encoder_in_feature_size"));
  const auto allow_remapping_by_area_matrix =
    this->declare_parameter<std::vector<int64_t>>("allow_remapping_by_area_matrix");
  const auto min_area_matrix = this->declare_parameter<std::vector<double>>("min_area_matrix");
  const auto max_area_matrix = this->declare_parameter<std::vector<double>>("max_area_matrix");

  detection_class_remapper_.setParameters(
    allow_remapping_by_area_matrix, min_area_matrix, max_area_matrix);

  {
    NMSParams p;
    p.nms_type_ = NMS_TYPE::IoU_BEV;
    p.target_class_names_ =
      this->declare_parameter<std::vector<std::string>>("iou_nms_target_class_names");
    p.search_distance_2d_ = this->declare_parameter<double>("iou_nms_search_distance_2d");
    p.iou_threshold_ = this->declare_parameter<double>("iou_nms_threshold");
    iou_bev_nms_.setParameters(p);
  }

  NetworkParam encoder_param(encoder_onnx_path, encoder_engine_path, trt_precision);
  NetworkParam head_param(head_onnx_path, head_engine_path, trt_precision);
  DensificationParam densification_param(
    densification_world_frame_id, densification_num_past_frames);

  if (point_cloud_range.size() != 6) {
    RCLCPP_WARN_STREAM(
      rclcpp::get_logger("lidar_centerpoint"),
      "The size of point_cloud_range != 6: use the default parameters.");
  }
  if (voxel_size.size() != 3) {
    RCLCPP_WARN_STREAM(
      rclcpp::get_logger("lidar_centerpoint"),
      "The size of voxel_size != 3: use the default parameters.");
  }
  CenterPointConfig config(
    class_names_.size(), point_feature_size, max_voxel_size, point_cloud_range, voxel_size,
    downsample_factor, encoder_in_feature_size, score_threshold, circle_nms_dist_threshold,
    yaw_norm_thresholds);
  detector_ptr_ =
    std::make_unique<CenterPointTRT>(encoder_param, head_param, densification_param, config);

  pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "~/input/pointcloud", rclcpp::SensorDataQoS{}.keep_last(1),
    std::bind(&LidarCenterPointNode::pointCloudCallback, this, std::placeholders::_1));
  objects_pub_ = this->create_publisher<autoware_auto_perception_msgs::msg::DetectedObjects>(
    "~/output/objects", rclcpp::QoS{1});

  // initialize debug tool
  {
    using tier4_autoware_utils::DebugPublisher;
    using tier4_autoware_utils::StopWatch;
    stop_watch_ptr_ = std::make_unique<StopWatch<std::chrono::milliseconds>>();
    debug_publisher_ptr_ = std::make_unique<DebugPublisher>(this, "lidar_centerpoint");
    stop_watch_ptr_->tic("cyclic_time");
    stop_watch_ptr_->tic("processing_time");
  }
}

void LidarCenterPointNode::pointCloudCallback(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr input_pointcloud_msg)
{
  const auto objects_sub_count =
    objects_pub_->get_subscription_count() + objects_pub_->get_intra_process_subscription_count();
  if (objects_sub_count < 1) {
    return;
  }

  if (stop_watch_ptr_) {
    stop_watch_ptr_->toc("processing_time", true);
  }

  std::vector<Box3D> det_boxes3d;
  bool is_success = detector_ptr_->detect(*input_pointcloud_msg, tf_buffer_, det_boxes3d);
  if (!is_success) {
    return;
  }

  std::vector<autoware_auto_perception_msgs::msg::DetectedObject> raw_objects;
  raw_objects.reserve(det_boxes3d.size());
  for (const auto & box3d : det_boxes3d) {
    autoware_auto_perception_msgs::msg::DetectedObject obj;
    box3DToDetectedObject(box3d, class_names_, has_twist_, obj);
    raw_objects.emplace_back(obj);
  }

  autoware_auto_perception_msgs::msg::DetectedObjects output_msg;
  output_msg.header = input_pointcloud_msg->header;
  output_msg.objects = iou_bev_nms_.apply(raw_objects);

  const double correction_min_length = 5.0;
  const auto voxelized_pcl_ptr = detector_ptr_->getVoxelizedPointCloudPtr();
  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_pcl_ptr(new pcl::PointCloud<pcl::PointXYZ>);
  downsampled_pcl_ptr->reserve(voxelized_pcl_ptr->size());
  pcl::VoxelGrid<pcl::PointXYZ> vg_filter;
  vg_filter.setInputCloud(voxelized_pcl_ptr);
  vg_filter.setLeafSize(0.32, 0.32, 4.0);
  vg_filter.filter(*downsampled_pcl_ptr);

  autoware_auto_perception_msgs::msg::DetectedObjects debug_raw_shape_objects_msg;
  debug_raw_shape_objects_msg.header = input_pointcloud_msg->header;
  pcl::PointCloud<pcl::PointXYZ>::Ptr debug_object_pc_pcl(new pcl::PointCloud<pcl::PointXYZ>);
  debug_object_pc_pcl->reserve(voxelized_pcl_ptr->size());
  auto valid_point = [](double p, double width) -> bool { return (p >= -width) && (p <= width); };
  auto get_sqr_dist = [](double a, double b) -> double { return a * a + b * b; };
  for (auto & object : output_msg.objects) {
    const auto label = perception_utils::getHighestProbLabel(object.classification);
    if (!perception_utils::isLargeVehicle(label)) {
      continue;
    }

    auto & dimensions = object.shape.dimensions;
    if (dimensions.x < correction_min_length) {
      continue;
    }

    auto & pose = object.kinematics.pose_with_covariance.pose;
    Eigen::Affine3d obj2base_transform;
    tf2::fromMsg(pose, obj2base_transform);

    double min_x_obj_coord(std::numeric_limits<double>::max()),
      max_x_obj_coord(std::numeric_limits<double>::min());
    for (const auto & pc : *downsampled_pcl_ptr) {
      const double sqr_dist_between_point_and_centroid =
        get_sqr_dist(pc.x - pose.position.x, pc.y - pose.position.y);
      const double sqrt_length_of_half_diagonal = get_sqr_dist(dimensions.x, dimensions.y);
      if (sqr_dist_between_point_and_centroid > sqrt_length_of_half_diagonal) {
        continue;
      }

      Eigen::Vector3d p_base_coord(pc.x, pc.y, pc.z);
      const auto p_obj_coord = obj2base_transform.inverse() * p_base_coord;
      if (!valid_point(p_obj_coord.x(), dimensions.x * 0.5f)) {
        continue;
      }
      if (!valid_point(p_obj_coord.y(), dimensions.y * 0.5f)) {
        continue;
      }
      if (!valid_point(p_obj_coord.z(), dimensions.z * 0.5f)) {
        continue;
      }

      if (p_obj_coord.x() < min_x_obj_coord) {
        min_x_obj_coord = p_obj_coord.x();
      }
      if (p_obj_coord.x() > max_x_obj_coord) {
        max_x_obj_coord = p_obj_coord.x();
      }

      debug_object_pc_pcl->emplace_back(pcl::PointXYZ(pc.x, pc.y, pc.z));
    }

    const double new_length = max_x_obj_coord - min_x_obj_coord;
    if (new_length <= correction_min_length) {
      const auto & width = dimensions.y;
      debug_raw_shape_objects_msg.objects.emplace_back(object);

      std::vector<Eigen::Vector3d> corners;
      corners.push_back(
        obj2base_transform * Eigen::Vector3d(dimensions.x * 0.5, dimensions.y * 0.5, 0.0));
      corners.push_back(
        obj2base_transform * Eigen::Vector3d(-dimensions.x * 0.5, dimensions.y * 0.5, 0.0));
      corners.push_back(
        obj2base_transform * Eigen::Vector3d(dimensions.x * 0.5, -dimensions.y * 0.5, 0.0));
      corners.push_back(
        obj2base_transform * Eigen::Vector3d(-dimensions.x * 0.5, -dimensions.y * 0.5, 0.0));

      const auto nearest_corner =
        *(std::min_element(corners.begin(), corners.end(), [](const auto & a, const auto & b) {
          return a.norm() < b.norm();
        }));

      Eigen::Vector3d nearest_corner_obj_coord = obj2base_transform.inverse() * nearest_corner;
      Eigen::Vector3d ex = Eigen::Vector3d(nearest_corner_obj_coord.x(), 0, 0).normalized();
      Eigen::Vector3d ey = Eigen::Vector3d(0, nearest_corner_obj_coord.y(), 0).normalized();
      Eigen::Vector3d new_position =
        nearest_corner -
        obj2base_transform.rotation() * (ex * correction_min_length * 0.5 + ey * width * 0.5);

      dimensions.x = correction_min_length;

      pose.position.x = new_position.x();
      pose.position.y = new_position.y();
    }
  }

  detection_class_remapper_.mapClasses(output_msg);

  if (objects_sub_count > 0) {
    objects_pub_->publish(output_msg);
  }

  // add processing time for debug
  if (debug_publisher_ptr_ && stop_watch_ptr_) {
    const double cyclic_time_ms = stop_watch_ptr_->toc("cyclic_time", true);
    const double processing_time_ms = stop_watch_ptr_->toc("processing_time", true);
    debug_publisher_ptr_->publish<tier4_debug_msgs::msg::Float64Stamped>(
      "debug/cyclic_time_ms", cyclic_time_ms);
    debug_publisher_ptr_->publish<tier4_debug_msgs::msg::Float64Stamped>(
      "debug/processing_time_ms", processing_time_ms);
  }

  if (debug_publisher_ptr_) {
    // const auto debug_pc_pcl = detector_ptr_->getVoxelizedPointCloudPtr();
    // if (debug_pc_pcl) {
    //   sensor_msgs::msg::PointCloud2 debug_pc_msg;
    //   pcl::toROSMsg(*debug_pc_pcl, debug_pc_msg);
    //   debug_pc_msg.header = input_pointcloud_msg->header;
    //   debug_publisher_ptr_->publish<sensor_msgs::msg::PointCloud2>(
    //     "debug/voxelized_pointcloud", debug_pc_msg);
    // }

    sensor_msgs::msg::PointCloud2 debug_object_pc_msg;
    pcl::toROSMsg(*debug_object_pc_pcl, debug_object_pc_msg);
    debug_object_pc_msg.header = input_pointcloud_msg->header;
    debug_publisher_ptr_->publish<sensor_msgs::msg::PointCloud2>(
      "debug/object_pointcloud", debug_object_pc_msg);

    debug_publisher_ptr_->publish<autoware_auto_perception_msgs::msg::DetectedObjects>(
      "debug/raw_shape_objects", debug_raw_shape_objects_msg);
  }
}

}  // namespace centerpoint

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(centerpoint::LidarCenterPointNode)
