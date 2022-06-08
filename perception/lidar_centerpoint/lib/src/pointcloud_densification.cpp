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

#include <pcl_ros/transforms.hpp>
#include <pointcloud_densification.hpp>
// #include <timer.hpp>

#include <boost/optional.hpp>

#include <pcl_conversions/pcl_conversions.h>
#ifdef ROS_DISTRO_GALACTIC
#include <tf2_eigen/tf2_eigen.h>
#else
#include <tf2_eigen/tf2_eigen.hpp>
#endif

#include <string>
#include <utility>

namespace
{
boost::optional<geometry_msgs::msg::Transform> getTransform(
  const tf2_ros::Buffer & tf_buffer, const std::string & target_frame_id,
  const std::string & source_frame_id, const rclcpp::Time & time)
{
  try {
    geometry_msgs::msg::TransformStamped transform_stamped;
    transform_stamped = tf_buffer.lookupTransform(
      target_frame_id, source_frame_id, time, rclcpp::Duration::from_seconds(0.5));
    return transform_stamped.transform;
  } catch (tf2::TransformException & ex) {
    RCLCPP_WARN_STREAM(rclcpp::get_logger("lidar_centerpoint"), ex.what());
    return boost::none;
  }
}

Eigen::Affine3f transformToEigen(const geometry_msgs::msg::Transform & t)
{
  Eigen::Affine3f a;
  a.matrix() = tf2::transformToEigen(t).matrix().cast<float>();
  return a;
}

}  // namespace

namespace centerpoint
{
PointCloudDensification::PointCloudDensification(const DensificationParam & param) : param_(param)
{
}

bool PointCloudDensification::enqueuePointCloud(
  const sensor_msgs::msg::PointCloud2 & pointcloud_msg, const tf2_ros::Buffer & tf_buffer)
{
  const auto header = pointcloud_msg.header;

  auto transform_world2current =
    getTransform(tf_buffer, header.frame_id, param_.world_frame_id(), header.stamp);
  if (!transform_world2current) {
    return false;
  }
  auto affine_world2current = transformToEigen(transform_world2current.get());

  // sensor_msgs::msg::PointCloud2 out_pc_msg;
  // sensor_msgs::PointCloud2Modifier pcd_modifier(out_pc_msg);
  // pcd_modifier.setPointCloud2FieldsByString(1, "xyz");
  // pcd_modifier.resize(total_point_size);
  // sensor_msgs::PointCloud2Iterator<float> x_out_iter(out_pc_msg, "x");
  // sensor_msgs::PointCloud2Iterator<float> y_out_iter(out_pc_msg, "y");
  // sensor_msgs::PointCloud2Iterator<float> z_out_iter(out_pc_msg, "z");

  // Timer timer;

  // std::size_t in_counter = 0;
  // std::size_t out_counter = 0;
  // for (sensor_msgs::PointCloud2ConstIterator<float> x_iter(pointcloud_msg, "x"),
  //      y_iter(pointcloud_msg, "y"), z_iter(pointcloud_msg, "z");
  //      x_iter != x_iter.end(); ++x_iter, ++y_iter, ++z_iter) {
  //   if ((*x_iter * *x_iter + *y_iter * *y_iter) < 100) {
  //     in_counter++;
  //   } else {
  //     out_counter++;
  //   }
  // }
  // std::cout << "in_counter " << in_counter << std::endl;
  // std::cout << "out_counter " << out_counter << std::endl;

  enqueue(pointcloud_msg, affine_world2current);
  dequeue();

  return true;
}

void PointCloudDensification::enqueue(
  const sensor_msgs::msg::PointCloud2 & msg, const Eigen::Affine3f & affine_world2current)
{
  affine_world2current_ = affine_world2current;
  current_timestamp_ = rclcpp::Time(msg.header.stamp).seconds();
  PointCloudWithTransform pointcloud = {msg, affine_world2current.inverse()};
  pointcloud_cache_.push_front(pointcloud);
}

void PointCloudDensification::dequeue()
{
  if (pointcloud_cache_.size() > param_.pointcloud_cache_size()) {
    pointcloud_cache_.pop_back();
  }
}

}  // namespace centerpoint
