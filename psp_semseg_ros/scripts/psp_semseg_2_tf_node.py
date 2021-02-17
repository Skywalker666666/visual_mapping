#!/usr/bin/env /home/zhiliu/anaconda3/envs/pspnet_kerastf/bin/python3

import os
import threading


import sys
#ros_path = '/opt/ros/melodic/lib/python2.7/dist-packages'
ros_path2 = '/home/zhiliu/Documents/catkin_ws_VoSM/devel/lib/python2.7/dist-packages'
#ros_path3 = '/home/zhiliu/Documents/Weapons/catkin_workspace_bridge/install/lib/python3.7/site-packages'
ros_path3 = '/home/zhiliu/Documents/Weapons/catkin_workspace_bridge_py3p5/install/lib/python3.5/site-packages'


#if ros_path in sys.path:
#    sys.path.remove(ros_path)

if ros_path2 in sys.path:
    sys.path.remove(ros_path2)

sys.path.insert(1,ros_path3)

import numpy as np
import cv2
from cv_bridge import CvBridge

import matplotlib.pyplot as plt


import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import RegionOfInterest


from os.path import splitext, join, isfile, isdir, basename
from scipy import misc, ndimage
from keras import backend as K
from keras.models import model_from_json, load_model
import tensorflow as tf

from glob import glob
from keras.utils.generic_utils import CustomObjectScope

import math

sys.path.append(ros_path2)


#from psp_semseg_ros.predict_semantic_single_img import SemanticSegmentation
from psp_semseg_ros.msg import Result
#from psp_semseg_ros import visualize
from psp_semseg_ros.pspnet_tf_2 import PSPNet50
from psp_semseg_ros.utils_2 import utils


from psp_semseg_ros.ade20k_labels_2 import ade20k_id2label


# Local path to trained weights file
#ROS_HOME = os.environ.get('ROS_HOME', os.path.join(os.environ['HOME'], '.ros'))

#PSP_TF_MODEL_PATH = os.path.join('/home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg/src/trainers/runs/coco/psp/default/', 'PSP_Resnet50_epoch2of2_auxon_batchs4_lr1en3.params')


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: CLASS_NAMES.index('teddy bear')
#CLASS_NAMES = ['banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 
#               'flower',
#               'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
#               'railroad',
#               'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick',
#               'wall-stone',
#               'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 
#               'fence-merged',
#               'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged',
#               'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged',
#               'food-other-merged',
#               'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']

# ID for stuff background
#CLASSES = [92, 93, 95, 100, 107, 109, 112, 118, 119, 122, 125, 128, 130, 133, 138, 141, 144, 145, 147, 148,
#           149, 151, 154, 155, 156, 159, 161, 166, 168, 171, 175, 176, 177, 178, 180, 181, 184, 185, 186,
#           187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200]
#

##class InferenceConfig():
##    # Set batch size to 1 since we'll be running inference on
##    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
##    GPU_COUNT = 1
##    IMAGES_PER_GPU = 1


class PSPSemSegNode(object):
    def __init__(self):
        self._cv_bridge = CvBridge()

        #config = InferenceConfig()
        #config.display()
 
        # GPU id 0: how many GPU we have for different machine:
        #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # CPU only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        self._visualization = rospy.get_param('~visualization', True)


        self._sess = tf.Session()
        K.set_session(self._sess)

        # Create model object for inference
        #with self._sess.as_default():
        self._model= PSPNet50(nb_classes=150, input_shape=(473, 473), weights='pspnet50_ade20k')

        # class name for all classes defined in dataset
        #self._classes_names = rospy.get_param('~class_names', CLASS_NAMES)
        #self._classes =  CLASSES
        self._last_msg = None
        self._msg_lock = threading.Lock()

        #self._class_colors = visualize.random_colors(len(CLASS_NAMES))

        self._publish_rate = rospy.get_param('~publish_rate', 100)

        

    def run(self):
        self._result_pub = rospy.Publisher('~psp_result', Result, queue_size=1)
        #vis_pub = rospy.Publisher('~visualization', Image, queue_size=1)
        rospy.Subscriber('~input', Image,
                         self._image_callback, queue_size=1)

        rate = rospy.Rate(self._publish_rate)
        img_cnt = 0
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
                rospy.loginfo("-------------------------------Received the image")
                img_cnt = img_cnt + 1
                print("img_cnt: " + str(img_cnt)) 
                # Run detection

                EVALUATION_SCALES = [1.0]
                

                #K.set_session(self._sess)
                with self._sess.as_default():
                    probs = self._model.predict_multi_scale(img=np_image, flip_evaluation=False, sliding_evaluation=False, scales=EVALUATION_SCALES)
                    rospy.loginfo("-------------------------------Detection performed")
                    # class map 
                    cm = np.argmax(probs, axis=2)
                    # convert class map to color map for visulizatoin
                    colored_class_image = utils.color_class_image(cm, 'pspnet50_ade20k')
                    # save color map
                    #filename = "/home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg_PSP2/data/predictions/test_result_for_vox_ros/pspnet_semseg_result_" + str(msg.header.stamp.to_sec()) + ".png"
                    #misc.imsave(filename, colored_class_image)
 

                #rospy.loginfo(str(cm.shape))

                result_msg = self._build_result_msg(msg, cm)
                self._result_pub.publish(result_msg)



                # Visualize results
                if self._visualization:
                    ##visualize_img = self._visualize(result, np_image, msg.header)

                    rospy.loginfo("Visualization is not implemented yet, but we saved rsult image in last function")
                    #cv_result = self._visualize_plt(result, np_image)
                    #image_msg = self._cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')
                    #vis_pub.publish(image_msg)

            rate.sleep()

    def _build_result_msg(self, msg, result):
        result_msg = Result()
        result_msg.header = msg.header
        predicted_categories = list(np.unique(result))

        for i, category in enumerate(predicted_categories):
            #print('Category: ')
            #print(category)
            #print("category_id: int(category) + 300 ")
            #print(str(category + 300))
            #print(ade20k_id2label[int(category)].name)
            if ade20k_id2label[int(category)].name == 'wall' or ade20k_id2label[int(category)].name == 'floor':
                print("category_id: int(category) + 300 ")
                print(str(category + 300))

                print(ade20k_id2label[int(category)].name)
                # offset to avoid the conflict of two dataset, coco and ade20k
                result_msg.class_ids.append(int(category) + 300)
                result_msg.class_names.append(ade20k_id2label[int(category)].name)
                mask = Image()
                mask.header = msg.header
                mask.height = result.shape[0]
                #print("height: " + str(mask.height))
                mask.width  = result.shape[1]
                #print("width: " + str(mask.width))
                mask.encoding = "mono8"
                mask.is_bigendian = False
                mask.step = mask.width
                binary_mask = (np.isin(result, category) * 1)
                uint8_binary_mask = binary_mask.astype(np.uint8)
                mask.data = (uint8_binary_mask * 255).tobytes()
                result_msg.masks.append(mask)

                #filename2 = "/home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg_PSP2/data/predictions/test_result_for_vox_ros/pspnet_semseg_result_seperate_mask_" + str(i) + "_"  + str(msg.header.stamp.to_sec()) + ".png"
                #misc.imsave(filename2, binary_mask * 255)
        return result_msg


#    def _visualize(self, result, image, msg_header):
#        visualize.show_semseg_result(result, msg_header, self._classes, self._classes_names)
#

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
