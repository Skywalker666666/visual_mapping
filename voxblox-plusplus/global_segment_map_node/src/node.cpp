// Copyright (c) 2019, ASL, ETH Zurich, Switzerland
// Licensed under the BSD 3-Clause License (see LICENSE for details)

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <ros/ros.h>

#include "voxblox_gsm/controller.h"
#include "voxblox_gsm/sliding_window_controller.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "gsm_node");

  //google::SetLogDestination(google::GLOG_INFO, "");
  // this one has to be called before the init
  FLAGS_log_dir = "/home/zhiliu/Documents/catkin_ws_VoSM_UPS/outputs/logs/";

  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();

  // The numbers of severity levels INFO, WARNING, ERROR, and FATAL are 0, 1, 2, and 3, respectively.
  //FLAGS_stderrthreshold = 1;
  FLAGS_stderrthreshold = 0;
  // not only log file also stderr
  FLAGS_alsologtostderr = true;
  // set the path for log files 

  ros::NodeHandle node_handle;
  ros::NodeHandle node_handle_private("~");


  voxblox::voxblox_gsm::Controller* controller;
  LOG(INFO) << "Starting Voxblox++ node.";
  controller = new voxblox::voxblox_gsm::Controller(&node_handle_private);

  ros::ServiceServer reset_map_srv;
  controller->advertiseResetMapService(&reset_map_srv);

  ros::ServiceServer toggle_integration_srv;
  controller->advertiseToggleIntegrationService(&toggle_integration_srv);

  ros::Subscriber segment_point_cloud_sub;
  controller->subscribeSegmentPointCloudTopic(&segment_point_cloud_sub);

  if (controller->publish_scene_mesh_) {
    controller->advertiseSceneMeshTopic();
    controller->advertiseSceneCloudTopic();
  }

  if (controller->compute_and_publish_bbox_) {
    controller->advertiseBboxTopic();
  }

  ros::ServiceServer generate_mesh_srv;
  controller->advertiseGenerateMeshService(&generate_mesh_srv);

  ros::ServiceServer get_scene_pointcloud;
  controller->advertiseGetScenePointcloudService(&get_scene_pointcloud);

  ros::ServiceServer save_segments_as_mesh_srv;
  controller->advertiseSaveSegmentsAsMeshService(&save_segments_as_mesh_srv);

  ros::ServiceServer extract_instances_srv;
  ros::ServiceServer get_list_semantic_instances_srv;
  ros::ServiceServer get_instance_bounding_box_srv;

  if (controller->enable_semantic_instance_segmentation_) {
    controller->advertiseExtractInstancesService(&extract_instances_srv);
    controller->advertiseGetListSemanticInstancesService(
        &get_list_semantic_instances_srv);
    controller->advertiseGetAlignedInstanceBoundingBoxService(
        &get_instance_bounding_box_srv);
  }

  // Spinner that uses a number of threads equal to the number of cores.
  ros::AsyncSpinner spinner(0);
  spinner.start();
  ros::waitForShutdown();

  LOG(INFO) << "Shutting down.";
  return 0;
}
