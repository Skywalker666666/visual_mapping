#!/usr/bin/env /home/zhiliu/anaconda3/envs/upsnet_pytorch/bin/python

import os
import threading
import sys
#ros_path = '/opt/ros/melodic/lib/python2.7/dist-packages'
ros_path2 = '/home/zhiliu/Documents/catkin_ws_VoSM_UPS/devel/lib/python2.7/dist-packages'
#ros_path3 = '/home/zhiliu/Documents/Weapons/catkin_workspace_bridge/install/lib/python3.7/site-packages'
#ros_path3 = '/home/zhiliu/Documents/Weapons/catkin_workspace_bridge_py3p5/install/lib/python3.5/site-packages'
ros_path4 = '/home/zhiliu/Documents/Weapons/catkin_workspace_bridge_py3p6/install/lib/python3.6/site-packages'

#if ros_path in sys.path:
#    sys.path.remove(ros_path)

if ros_path2 in sys.path:
    sys.path.remove(ros_path2)

sys.path.insert(1,ros_path4)

import torch
import torch.nn as nn
import argparse

import numpy as np
import cv2
from cv_bridge import CvBridge


import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import RegionOfInterest


from scipy import misc, ndimage
#from keras import backend as K
#from keras.models import model_from_json, load_model
#import tensorflow as tf
#from keras.utils.generic_utils import CustomObjectScope

import torch
import torch.nn as nn

import math

sys.path.append(ros_path2)

src_sys2 = os.path.dirname(__file__)
print(src_sys2)
#sys.path.insert(1, src_sys2)
#sys.path.insert(1, os.path.join(src_sys2, '../src/'))
sys.path.insert(1, os.path.join(src_sys2, '../src/'))
sys.path.insert(1, os.path.join(src_sys2, '../src/upsnet'))
sys.path.insert(1, os.path.join(src_sys2, '../../'))
print(sys.path)


from upsnet.config.config import *
from upsnet.config.parse_args import parse_args
from upsnet.models import *
from upsnet.dataset.base_dataset_special import *
#from PIL import Image as PILImage
# if we need to use this, we need use it as PILImage
from ups_panseg_ros.msg import Result




# Local path to trained weights file
#ROS_HOME = os.environ.get('ROS_HOME', os.path.join(os.environ['HOME'], '.ros'))

#PSP_TF_MODEL_PATH = os.path.join('/home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg/src/trainers/runs/coco/psp/default/', 'PSP_Resnet50_epoch2of2_auxon_batchs4_lr1en3.params')


class UPSPanSegNode(object):
    def __init__(self):
        self._cv_bridge = CvBridge()

        #config = InferenceConfig()
        #config.display()
 
        self._visualization = rospy.get_param('~visualization', True)

        parser = argparse.ArgumentParser()
        args, rest = parser.parse_known_args()
        args.cfg = "/home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/upsnet/experiments/upsnet_resnet50_coco_solo_1gpu.yaml"

        args.weight_path = "/home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/output/upsnet/coco/upsnet_resnet50_coco_1gpu/train2017/upsnet_resnet_50_coco_234000.pth"
        
        #args.cfg = "/home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/upsnet/experiments/upsnet_resnet50_coco_4gpu.yaml"
        
        #args.weight_path = "/home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/UnifiedPanopticSeg/UPSNet_PanMapping/model/upsnet_resnet_50_coco_90000.pth"
        
        args.eval_only = False
        
        update_config(args.cfg)



        # Create model object for inference
        #with self._sess.as_default():
        self._model= eval("resnet_50_upsnet")().cuda()
        self._model.load_state_dict(torch.load(args.weight_path))
        for p in self._model.parameters():
            p.requires_grad = False
        
        
        self._model.eval()

        # class name for all classes defined in dataset
        #self._classes_names = rospy.get_param('~class_names', CLASS_NAMES)
        #self._classes =  CLASSES
        self._last_msg = None
        self._msg_lock = threading.Lock()

        #self._class_colors = visualize.random_colors(len(CLASS_NAMES))

        self._publish_rate = rospy.get_param('~publish_rate', 100)
   

    def run(self):
        self._result_pub = rospy.Publisher('~ups_result', Result, queue_size=1)
        #vis_pub = rospy.Publisher('~visualization', Image, queue_size=1)
        rospy.Subscriber('~input', Image,
                         self._image_callback, queue_size=1)

        rate = rospy.Rate(self._publish_rate)
        img_cnt = 0

        target_size = config.test.scales[0]

         

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
                rospy.loginfo("-------------------------------Received the image for ups module")
                img_cnt = img_cnt + 1
                print("img_cnt: " + str(img_cnt)) 

                # Run detection
                #np_image = cv2.imread("/home/zhiliu/.mxnet/datasets/coco/val2017/000000022396.jpg")
                ims, im_scales = prep_im_for_blob(np_image, config.network.pixel_means, [target_size], config.test.max_size)
                im_trans = ims[0].transpose(2, 0, 1)
                
                im_infos = np.array([[
                   ims[0].shape[0],
                   ims[0].shape[1],
                   im_scales[0]]], np.float32)
                
                batch = [{'data': im_trans , 'im_info' : im_infos}]
                blob = {}
                
                for key in batch[0]:
                    if key == 'data':
                        blob.update({'data': torch.from_numpy(im_list_to_blob([b['data'] for b in batch]))})
                        if config.network.has_panoptic_head:
                            blob.update({'data_4x': torch.from_numpy(im_list_to_blob([b['data'] for b in batch], scale=1/4.))})
                    elif key == 'im_info':
                        blob.update({'im_info': np.vstack([b['im_info'] for b in batch])})
                
                for k, v in blob.items():
                    # 'data' and 'data_4x' are cuda type, img_info is non cuda
                    blob[k] = v.cuda() if torch.is_tensor(v) else v
                
                batch_clt = blob
                
                with torch.no_grad():
                    output_all = self._model(batch_clt)
                
                output = {k: v.data.cpu().numpy() for k, v in output_all.items()}
                
                sseg = output['fcn_outputs']
                pano = output['panoptic_outputs']
                pano_cls_ind = output['panoptic_cls_inds']
                
                ssegs = []
                #for i, sseg in enumerate(output['ssegs']):
                sseg = sseg.squeeze(0).astype('uint8')[:int(im_infos[0][0]), :int(im_infos[0][1])]
                ssegs.append(cv2.resize(sseg, None, None, fx=1/im_infos[0][2], fy=1/im_infos[0][2], interpolation=cv2.INTER_NEAREST))
                
                panos = []
                pano = pano.squeeze(0).astype('uint8')[:int(im_infos[0][0]), :int(im_infos[0][1])]
                #panos = cv2.resize(pano, None, None, fx=1/im_infos[0][2], fy=1/im_infos[0][2], interpolation=cv2.INTER_NEAREST)
                panos.append(cv2.resize(pano, None, None, fx=1/im_infos[0][2], fy=1/im_infos[0][2], interpolation=cv2.INTER_NEAREST))
                
                pano_cls_inds = []
                pano_cls_inds.append(pano_cls_ind)                 
                pred_pans_2ch = get_unified_pan_result(ssegs, panos, pano_cls_inds, stuff_area_limit=4 * 64 * 64)
                
                print("unique label in final 2ch: ")
                print("pan_seg: ")
                print(np.unique(pred_pans_2ch[0][:,:,0]))
                print("pan_ins: ")
                print(np.unique(pred_pans_2ch[0][:,:,1]))
                
                    #probs = self._model.predict_multi_scale(img=np_image, flip_evaluation=False, sliding_evaluation=False, scales=EVALUATION_SCALES)
                    #rospy.loginfo("-------------------------------Detection performed")
                    # class map 
                    #cm = np.argmax(probs, axis=2)
                    # convert class map to color map for visulizatoin
                    #colored_class_image = utils.color_class_image(cm, 'pspnet50_ade20k')
                    #filename = "/home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg_PSP2/data/predictions/test_result_for_vox_ros/pspnet_semseg_result_" + str(msg.header.stamp.to_sec()) + ".png"
                    #misc.imsave(filename, colored_class_image)
                #rospy.loginfo(str(cm.shape))
               
                # one image per round, so [0], cuz pred_pans_2ch is a list, like[[]]
                result_msg = self._build_result_msg(msg, pred_pans_2ch[0])
                self._result_pub.publish(result_msg)
                
                # Visualize results
                #if self._visualization:
                    ##visualize_img = self._visualize(result, np_image, msg.header)
                
                    #rospy.loginfo("Visualization is not implemented yet, but we saved rsult image in last function")
                    #cv_result = self._visualize_plt(result, np_image)
                    #image_msg = self._cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')
                    #vis_pub.publish(image_msg)
            rate.sleep()

    def _build_result_msg(self, msg, result):
        result_msg = Result()
        result_msg.header = msg.header
        #segments categories:
        #predicted_categories = list(np.unique(result[:,:,0]))
        #predicted_instances  = list(np.unique(result[:,:,1]))
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # convert to np.uint32 is important, because 25 * 1000 is maximum
        pan_2ch = np.uint32(result)
        OFFSET = 1000
        VOID   = 255
        pan = OFFSET * pan_2ch[:, :, 0] + pan_2ch[:, :, 1]
        l = np.unique(pan)
        print("pan: ")
        print(np.unique(pan))
        #differentiate different instances with same categories.
        for el in l:
            # floor division, very cool
            sem_mask = (pan == el)

            category = el // OFFSET
            if category == VOID:
                continue
            # id 42: floor_merged
            # id 51: wall_other_merged
            # id 52: rug_merged
            print("py py py categories: ")
            print(category)

            #if int(category) >= 53 or int(category) == 41 or int(category) == 42 or int(category) == 51 or #int(category) == 52:
            #print("size of mask: " + str(sum(sum(sem_mask*1))))
            if int(category) >= 1 and sum(sum(sem_mask*1)) > 20:
                # category = 0 is banner, but looks like it is a bug.
                #print("category_id: coco panoptic categories annotation")
                #print(" sem: " + str(category) + " of pan: " + str(el))
                #print("size of mask: " + str(sum(sum(sem_mask*1))))
                
                # watch out, it is uin8 for id in voxblox <= 255
                result_msg.class_ids.append(np.uint8(category))
                #result_msg.class_names.append()
                mask = Image()
                mask.header = msg.header
                mask.height = result[:,:,0].shape[0]
                #print("height: " + str(mask.height))
                mask.width  = result[:,:,0].shape[1]
                #print("width: " + str(mask.width))
                mask.encoding = "mono8"
                mask.is_bigendian = False
                mask.step = mask.width
                binary_mask = sem_mask * 1
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
    rospy.init_node('ups_panseg')

    node = UPSPanSegNode()
    node.run()

if __name__ == '__main__':
    main()
