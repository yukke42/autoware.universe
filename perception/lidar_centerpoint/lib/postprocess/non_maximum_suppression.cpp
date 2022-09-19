// Copyright 2022 TIER IV, Inc.
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

#include "lidar_centerpoint/postprocess/non_maximum_suppression.hpp"

#include "perception_utils/geometry.hpp"
#include "perception_utils/perception_utils.hpp"
#include "tier4_autoware_utils/tier4_autoware_utils.hpp"

#include <Eigen/Eigen>

namespace centerpoint
{

void NonMaximumSuppression::setParams(const NMSParams & params) { params_ = params; }

std::vector<DetectedObject> NonMaximumSuppression::apply(
  const std::vector<DetectedObject> & input_objects)
{
  const auto search_sqr_dist_2d = params_.search_distance_2d_ * params_.search_distance_2d_;
  // NOTE(yukke42): row = target objects to be suppressed, col = source objects to be compared
  Eigen::MatrixXd lower_tri_mat = Eigen::MatrixXd::Zero(input_objects.size(), input_objects.size());
  for (std::size_t target_i = 0; target_i < input_objects.size(); ++target_i) {
    for (std::size_t source_i = 0; source_i < target_i; ++source_i) {
      const auto & target_obj = input_objects.at(target_i);
      const auto & source_obj = input_objects.at(source_i);
      const auto target_label = perception_utils::getHighestProbLabel(target_obj.classification);
      const auto source_label = perception_utils::getHighestProbLabel(source_obj.classification);
      if (!isVehicleLabel(target_label) || !isVehicleLabel(source_label)) {
        continue;
      }

      const auto sqrt_dist_2d = tier4_autoware_utils::calcSquaredDistance2d(
        perception_utils::getPose(target_obj), perception_utils::getPose(source_obj));
      if (sqrt_dist_2d > search_sqr_dist_2d) {
        continue;
      }

      if (params_.nms_type_ == NMS_TYPE::IoU_BEV) {
        const double iou = perception_utils::get2dIoU(target_obj, source_obj);
        lower_tri_mat(target_i, source_i) = iou;
        // NOTE(yukke42): If the target object has any objects with iou > iou_threshold, it
        // will be suppressed regardless of later results.
        if (iou > params_.iou_threshold_) {
          break;
        }
      }
    }
  }

  std::vector<DetectedObject> output_objects;
  output_objects.reserve(input_objects.size());
  for (std::size_t i = 0; i < input_objects.size(); ++i) {
    const auto value = lower_tri_mat.row(i).maxCoeff();
    if (params_.nms_type_ == NMS_TYPE::IoU_BEV) {
      if (value <= params_.iou_threshold_) {
        output_objects.emplace_back(input_objects.at(i));
      }
    }
  }

  return output_objects;
}
}  // namespace centerpoint
