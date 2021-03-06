#!/usr/bin/env python2
import os
import threading
import numpy as np
import cv2
from cv_bridge import CvBridge
import matplotlib.pyplot as plt

import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import RegionOfInterest

from mask_rcnn_ros import coco
from mask_rcnn_ros import utils
from mask_rcnn_ros import model as modellib
from mask_rcnn_ros import visualize
from mask_rcnn_ros.msg import Result


# Local path to trained weights file
ROS_HOME = os.environ.get('ROS_HOME', os.path.join(os.environ['HOME'], '.ros'))
COCO_MODEL_PATH = os.path.join(ROS_HOME, 'mask_rcnn_coco.h5')

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: CLASS_NAMES.index('teddy bear')
CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class MaskRCNNNode(object):
    def __init__(self):
        self._cv_bridge = CvBridge()

        config = InferenceConfig()
        config.display()

        self._visualization = rospy.get_param('~visualization', True)

        # Create model object in inference mode.
        #self._model = modellib.MaskRCNN(mode="inference", model_dir="",
                                        #config=config)
        # Load weights trained on MS-COCO
        model_path = rospy.get_param('~model_path', COCO_MODEL_PATH)
        # Download COCO trained weights from Releases if needed
        #if model_path == COCO_MODEL_PATH and not os.path.exists(COCO_MODEL_PATH):
            #utils.download_trained_weights(COCO_MODEL_PATH)

        #self._model.load_weights(model_path, by_name=True)

        self._class_names = rospy.get_param('~class_names', CLASS_NAMES)

        self._last_msg = None
        self._msg_lock = threading.Lock()

        self._class_colors = visualize.random_colors(len(CLASS_NAMES))

        self._publish_rate = rospy.get_param('~publish_rate', 100)

    def run(self):
        self._result_pub = rospy.Publisher('~result', Result, queue_size=1)
        vis_pub = rospy.Publisher('~visualization', Image, queue_size=1)
        rospy.Subscriber('~input', Image,
                         self._image_callback, queue_size=1)

        rate = rospy.Rate(self._publish_rate)
        while not rospy.is_shutdown():
            if self._msg_lock.acquire(False):
                msg = self._last_msg
                self._last_msg = None
                self._msg_lock.release()
            else:
                rate.sleep()
                continue

            if msg is not None:
                #np_image = self._cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
                if msg.header.stamp.to_sec() < 0.5:
                    print("^^^^^^^^^^^^^^^^^^Get image 1")
                    print(str(msg.header.stamp.to_sec()))
                    img_cityscape = "/home/zhiliu/Documents/Panoptic_Segement/Videopanoptic/VideoPanopticSeg/work_dirs/cityscapes_vps/fusetrack_vpct/test_pans_unified/pan_pred/0500_2995_munster_000173_000004.png"
                elif 0.5 <= msg.header.stamp.to_sec() < 2:
                    print("^^^^^^^^^^^^^^^^^^Get image 2")
                    print(str(msg.header.stamp.to_sec()))
                    img_cityscape = "/home/zhiliu/Documents/Panoptic_Segement/Videopanoptic/VideoPanopticSeg/work_dirs/cityscapes_vps/fusetrack_vpct/test_pans_unified/pan_pred/0500_2996_munster_000173_000009.png"
                elif 2 <= msg.header.stamp.to_sec() < 3.9:
                    print("^^^^^^^^^^^^^^^^^^Get image 3")
                    print(str(msg.header.stamp.to_sec()))
                    img_cityscape = "/home/zhiliu/Documents/Panoptic_Segement/Videopanoptic/VideoPanopticSeg/work_dirs/cityscapes_vps/fusetrack_vpct/test_pans_unified/pan_pred/0500_2997_munster_000173_000014.png"
                else:
                    print("^^^^^^^^^^^^^^^^^^Get image 4")
                    print(str(msg.header.stamp.to_sec()))
                    img_cityscape = "/home/zhiliu/Documents/Panoptic_Segement/Videopanoptic/VideoPanopticSeg/work_dirs/cityscapes_vps/fusetrack_vpct/test_pans_unified/pan_pred/0500_2998_munster_000173_000019.png"






            
                np_image = np.uint32(cv2.imread(img_cityscape))
 
                # Run detection
                print("Run detection")
                #results = self._model.detect([np_image], verbose=0)
                OFFSET = 256
                #BGR, yes, verified, here it is BGR
                result = np_image[:, :, 0] + OFFSET * np_image[:, :, 1] + OFFSET * OFFSET * np_image[:, :, 2] 
                result_msg = self._build_result_msg(msg, result)
                self._result_pub.publish(result_msg)

                # Visualize results
                #if self._visualization:
                    #cv_result = self._visualize_plt(result, np_image)
                    #image_msg = self._cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')
                    #vis_pub.publish(image_msg)

            rate.sleep()

    def _build_result_msg(self, msg, result):
        # RGB color code for Cityscape semantic class
        # 0 - 10
        cityscapes_vps_color =[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], 
                               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], 
                               [107, 142, 35], [152, 251, 152], [70, 130, 180]]
        
        # 11, 12, 13, 14
        cisc_vps_munster173_instance_colr = [[0,0,142],[0,6,155],[6,20,134],[24,0,156]]
        
        result_msg = Result()
        result_msg.header = msg.header
        
        l = np.unique(result)
        
        for el in l:
            # add it for semantic mapping, remove it for pcd file
            if el == 180 + 256 * 130 + 256 * 256 * 70:
                # sky: color[]
                print("sky is here^^^^^^^^^^^^^^^^^^^^: ")
                continue

            if el == 142 + 256 * 0 + 256 * 256 * 0:
                # car0: color[]
                continue

            if el == 155 + 256 * 6 + 256 * 256 * 0:
                # car0: color[]
                continue

            if el == 134 + 256 * 20 + 256 * 256 * 6:
                # car0: color[]
                continue

            if el == 156 + 256 * 0 + 256 * 256 * 24:
                # car0: color[]
                continue
            
            sem_mask = (result == el) * 1 
            #if np.sum(sem_mask) > 64 * 64:
            
            if np.sum(sem_mask) > 4 * 4:
                C1 = el % 256
                C2 = (el // 256) % 256
                C3 = ((el // 256) // 256)
                #print("C1: " + str(C1))
                #print("C2: " + str(C2))
                #print("C3: " + str(C3))
                sem_id = 0
                for i,ele in enumerate(cityscapes_vps_color):
                    # BGR and RGB
                    if [C3,C2, C1] == ele:
                        sem_id = i
                        break
                for j,ele2 in enumerate(cisc_vps_munster173_instance_colr):
                    if [C3,C2,C1]  == ele2:
                        sem_id = j + 11
                        break
     
                #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^sem_id: " + str(sem_id))
                result_msg.class_ids.append(np.uint8(sem_id))
                mask = Image()
                mask.header = msg.header
                mask.height = result.shape[0]
                #print("height: " + str(mask.height))
                mask.width  = result.shape[1]

                mask.encoding = "mono8"
                mask.step = mask.width
                binary_mask = sem_mask
                uint8_binary_mask = binary_mask.astype(np.uint8)
                mask.data = (uint8_binary_mask * 255).tobytes()
                result_msg.masks.append(mask)
        return result_msg

    def _visualize(self, result, image):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        axes = fig.gca()
        visualize.display_instances(image, result['rois'], result['masks'],
                                    result['class_ids'], CLASS_NAMES,
                                    result['scores'], ax=axes,
                                    class_colors=self._class_colors)
        fig.tight_layout()
        canvas.draw()
        result = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

        _, _, w, h = fig.bbox.bounds
        result = result.reshape((int(h), int(w), 3))
        return result

    def _get_fig_ax(self):
        """Return a Matplotlib Axes array to be used in
        all visualizations. Provide a
        central point to control graph sizes.

        Change the default size attribute to control the size
        of rendered images
        """
        fig, ax = plt.subplots(1)
        plt.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        return fig, ax

    def _visualize_plt(self, result, image):
        fig, ax = self._get_fig_ax()
        image = visualize.display_instances_plt(
            image,
            result['rois'],
            result['masks'],
            result['class_ids'],
            CLASS_NAMES,
            result['scores'],
            fig=fig,
            ax=ax)

        return image

    def _image_callback(self, msg):
        rospy.logdebug("Get an image")
        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._msg_lock.release()


def main():
    rospy.init_node('mask_rcnn', log_level=rospy.INFO)

    node = MaskRCNNNode()
    node.run()


if __name__ == '__main__':
    main()
