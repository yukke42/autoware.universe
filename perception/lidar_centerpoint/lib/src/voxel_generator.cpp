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

#include <timer.hpp>
#include <voxel_generator.hpp>

#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace centerpoint
{
VoxelGeneratorTemplate::VoxelGeneratorTemplate(const DensificationParam & param)
{
  pd_ptr_ = std::make_unique<PointCloudDensification>(param);
}

bool VoxelGeneratorTemplate::enqueuePointCloud(
  const sensor_msgs::msg::PointCloud2 & input_pointcloud_msg, const tf2_ros::Buffer & tf_buffer)
{
  return pd_ptr_->enqueuePointCloud(input_pointcloud_msg, tf_buffer);
}

std::size_t VoxelGenerator::pointsToVoxels(
  std::vector<float> & voxels, std::vector<int> & coordinates,
  std::vector<float> & num_points_per_voxel)
{
  // voxels (float): (max_num_voxels * max_num_points_per_voxel * point_feature_size)
  // coordinates (int): (max_num_voxels * point_dim_size)
  // num_points_per_voxel (float): (max_num_voxels)

  Timer timer;
  const std::size_t grid_size = Config::grid_size_z * Config::grid_size_y * Config::grid_size_x;
  std::vector<int> coord_to_voxel_idx(grid_size, -1);

  std::size_t voxel_cnt = 0;  // @return
  std::array<float, Config::point_feature_size> point;
  std::array<float, Config::point_dim_size> coord_zyx;
  bool out_of_range;
  std::size_t point_cnt;
  int c, coord_idx, voxel_idx;
  float * in_point;
  Eigen::Vector3f point_current, point_past;

  std::vector<std::size_t> iterator(250000);
  std::iota(iterator.begin(), iterator.end(), 0);
  std::random_shuffle(iterator.begin(), iterator.end());
  std::cout << "random_shuffle " << timer.toc() << std::endl;

  for (auto pc_cache_iter = pd_ptr_->getPointCloudCacheIter(); !pd_ptr_->isCacheEnd(pc_cache_iter);
       pc_cache_iter++) {
    auto & pc_msg = pc_cache_iter->pointcloud_msg;
    auto affine_past2current =
      pd_ptr_->getAffineWorldToCurrent() * pc_cache_iter->affine_past2world;
    float timelag = static_cast<float>(
      pd_ptr_->getCurrentTimestamp() - rclcpp::Time(pc_msg.header.stamp).seconds());

    const auto data_size = pc_msg.data.size();
    const auto point_step = pc_msg.point_step;
    const auto point_size = data_size / point_step;
    for (const auto & iter : iterator) {
      if (iter >= point_size) {
        continue;
      }
      in_point = reinterpret_cast<float *>(&pc_msg.data[iter * point_step]);
      point_past << in_point[0], in_point[1], in_point[2];
      point_current = affine_past2current * point_past;

      point[0] = point_current.x();
      point[1] = point_current.y();
      point[2] = point_current.z();
      point[3] = timelag;

      out_of_range = false;
      for (std::size_t di = 0; di < Config::point_dim_size; di++) {
        c = static_cast<int>((point[di] - range_[di]) * recip_voxel_size_[di]);
        if (c < 0 || c >= grid_size_[di]) {
          out_of_range = true;
          break;
        }
        coord_zyx[Config::point_dim_size - di - 1] = c;
      }
      if (out_of_range) {
        continue;
      }

      coord_idx = coord_zyx[0] * Config::grid_size_y * Config::grid_size_x +
                  coord_zyx[1] * Config::grid_size_x + coord_zyx[2];
      voxel_idx = coord_to_voxel_idx[coord_idx];
      if (voxel_idx == -1) {
        voxel_idx = voxel_cnt;
        if (voxel_cnt >= Config::max_num_voxels) {
          continue;
        }

        voxel_cnt++;
        coord_to_voxel_idx[coord_idx] = voxel_idx;
        for (std::size_t di = 0; di < Config::point_dim_size; di++) {
          coordinates[voxel_idx * Config::point_dim_size + di] = coord_zyx[di];
        }
      }

      point_cnt = num_points_per_voxel[voxel_idx];
      if (point_cnt < Config::max_num_points_per_voxel) {
        for (std::size_t fi = 0; fi < Config::point_feature_size; fi++) {
          voxels
            [voxel_idx * Config::max_num_points_per_voxel * Config::point_feature_size +
             point_cnt * Config::point_feature_size + fi] = point[fi];
        }
        num_points_per_voxel[voxel_idx]++;
      }
    }
  }

  std::cout << timer.toc() << std::endl;

  return voxel_cnt;
}

}  // namespace centerpoint
