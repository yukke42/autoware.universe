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

#ifndef LIDAR_CENTERPOINT__POSTPROCESS__NON_MAXIMUM_SUPPRESSION_HPP_
#define LIDAR_CENTERPOINT__POSTPROCESS__NON_MAXIMUM_SUPPRESSION_HPP_

#include <autoware_auto_perception_msgs/msg/detected_object.hpp>

#include <vector>

namespace centerpoint
{
using autoware_auto_perception_msgs::msg::DetectedObject;

// TODO(yukke42): move to the shared library
inline bool isVehicleLabel(const std::uint8_t label)
{
  using Label = autoware_auto_perception_msgs::msg::ObjectClassification;
  return label == Label::CAR || label == Label::BUS || label == Label::TRUCK ||
         label == Label::TRAILER;
}

// TODO(yukke42): now only support IoU_BEV
enum class NMS_TYPE {
  IoU_BEV
  // IoU_3D
  // Distance_2D
  // Distance_3D
};

struct NMSParams
{
  NMS_TYPE nms_type_{};
  double search_distance_2d_{};
  double iou_threshold_{};
  double distance_threshold_{};
};

class NonMaximumSuppression
{
public:
  void setParams(const NMSParams & params);

  std::vector<DetectedObject> apply(const std::vector<DetectedObject> & input_objects);

private:
  NMSParams params_{};
};

}  // namespace centerpoint

#endif  // LIDAR_CENTERPOINT__POSTPROCESS__NON_MAXIMUM_SUPPRESSION_HPP_
