#include "global_feature_map/feature_types.h"

namespace voxblox {

template <>
std::string getFeatureType<Feature3D>() {
  return "Feature3D";
}

}  // namespace voxblox
