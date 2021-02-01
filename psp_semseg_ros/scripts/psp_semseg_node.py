#!/usr/bin/env /home/zhiliu/anaconda3/envs/panoptic_segmentation/bin/python
import os
import threading
import numpy as np

import sys
#ros_path = '/opt/ros/melodic/lib/python2.7/dist-packages'

ros_path2 = '/home/zhiliu/Documents/catkin_ws_VoSM/devel/lib/python2.7/dist-packages'
ros_path3 = '/home/zhiliu/Documents/Weapons/catkin_workspace_bridge/install/lib/python3.7/site-packages'
#if ros_path in sys.path:
#    sys.path.remove(ros_path)


if ros_path2 in sys.path:
    sys.path.remove(ros_path2)

sys.path.insert(1,ros_path3)

import cv2
from cv_bridge import CvBridge

import matplotlib.pyplot as plt

import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import RegionOfInterest

#from mask_rcnn_ros import coco
#from mask_rcnn_ros import utils
#from mask_rcnn_ros import model as modellib
#from mask_rcnn_ros import visualize
#from mask_rcnn_ros.msg import Result


sys.path.append(ros_path2)

from psp_semseg_ros.predict_semantic_single_img import SemanticSegmentation

import mxnet as mx


# Local path to trained weights file
#ROS_HOME = os.environ.get('ROS_HOME', os.path.join(os.environ['HOME'], '.ros'))
#COCO_MODEL_PATH = os.path.join(ROS_HOME, 'mask_rcnn_coco.h5')

PSP_MODEL_PATH = os.path.join('/home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg/src/trainers/runs/coco/psp/default/', 'PSP_Resnet50_epoch2of2_auxon_batchs4_lr1en3.params')


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: CLASS_NAMES.index('teddy bear')
CLASS_NAMES = ['banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 
               'flower',
               'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
               'railroad',
               'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick',
               'wall-stone',
               'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 
               'fence-merged',
               'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged',
               'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged',
               'food-other-merged',
               'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']

# ID for stuff background
CLASSES = [92, 93, 95, 100, 107, 109, 112, 118, 119, 122, 125, 128, 130, 133, 138, 141, 144, 145, 147, 148,
           149, 151, 154, 155, 156, 159, 161, 166, 168, 171, 175, 176, 177, 178, 180, 181, 184, 185, 186,
           187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200]


'''
class InferenceConfig():
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
'''

class PSPSemSegNode(object):
    def __init__(self):
        self._cv_bridge = CvBridge()

        #config = InferenceConfig()
        #config.display()

        self._visualization = rospy.get_param('~visualization', True)

        # Create model object in inference mode.
        self._model = SemanticSegmentation(model_path=PSP_MODEL_PATH, no_cuda=False)



        self._class_names = rospy.get_param('~class_names', CLASS_NAMES)

        self._last_msg = None
        self._msg_lock = threading.Lock()

        #self._class_colors = visualize.random_colors(len(CLASS_NAMES))

        self._publish_rate = rospy.get_param('~publish_rate', 100)

    def run(self):
        #self._result_pub = rospy.Publisher('~result', Result, queue_size=1)
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
                np_image = self._cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
                #np_image = []
                # Run detection
                print("Run detection")
                result = self._model.predict(mx.nd.array(np_image))

                rospy.loginfo("-------------------------------Received the image:")
                rospy.loginfo(str(result.shape))

                #result_msg = self._build_result_msg(msg, result)
                #self._result_pub.publish(result_msg)

                # Visualize results
                if self._visualization:
                    rospy.loginfo("Visualization is not implemented yet.")
                    #cv_result = self._visualize_plt(result, np_image)
                    #image_msg = self._cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')
                    #vis_pub.publish(image_msg)

            rate.sleep()

#    def _build_result_msg(self, msg, result):
#        result_msg = Result()
#        result_msg.header = msg.header
#        for i, (y1, x1, y2, x2) in enumerate(result['rois']):
#            box = RegionOfInterest()
#            box.x_offset = np.asscalar(x1)
#            box.y_offset = np.asscalar(y1)
#            box.height = np.asscalar(y2 - y1)
#            box.width = np.asscalar(x2 - x1)
#            result_msg.boxes.append(box)
#
#            class_id = result['class_ids'][i]
#            result_msg.class_ids.append(class_id)
#
#            class_name = self._class_names[class_id]
#            result_msg.class_names.append(class_name)
#
#            score = result['scores'][i]
#            result_msg.scores.append(score)
#
#            mask = Image()
#            mask.header = msg.header
#            mask.height = result['masks'].shape[0]
#            mask.width = result['masks'].shape[1]
#            mask.encoding = "mono8"
#            mask.is_bigendian = False
#            mask.step = mask.width
#            mask.data = (result['masks'][:, :, i] * 255).tobytes()
#            result_msg.masks.append(mask)
#        return result_msg

#    def _visualize(self, result, image):
#        from matplotlib.backends.backend_agg import FigureCanvasAgg
#        from matplotlib.figure import Figure
#
#        fig = Figure()
#        canvas = FigureCanvasAgg(fig)
#        axes = fig.gca()
#        visualize.display_instances(image, result['rois'], result['masks'],
#                                    result['class_ids'], CLASS_NAMES,
#                                    result['scores'], ax=axes,
#                                    class_colors=self._class_colors)
#        fig.tight_layout()
#        canvas.draw()
#        result = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
#
#        _, _, w, h = fig.bbox.bounds
#        result = result.reshape((int(h), int(w), 3))
#        return result

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

#    def _visualize_plt(self, result, image):
#        fig, ax = self._get_fig_ax()
#        image = visualize.display_instances_plt(
#            image,
#            result['rois'],
#            result['masks'],
#            result['class_ids'],
#            CLASS_NAMES,
#            result['scores'],
#            fig=fig,
#            ax=ax)
#
#        return image

    def _image_callback(self, msg):
        rospy.logdebug("Get an image")
        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._msg_lock.release()

def main():
    rospy.init_node('psp_semseg')

    node = PSPSemSegNode()
    node.run()

if __name__ == '__main__':
    main()
