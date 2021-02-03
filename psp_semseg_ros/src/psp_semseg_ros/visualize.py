#!/usr/bin/env /home/zhiliu/anaconda3/envs/panoptic_segmentation/bin/python

import os
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import PIL.Image     as Image


import json
from collections import defaultdict
try:
    from pycocotools import mask as COCOmask
except Exception:
    raise Exception("Please install pycocotools module from https://github.com/cocodataset/cocoapi")


import sys
sys.path.insert(1, '/home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg/src/panopticapi/')

from panopticapi.utils import IdGenerator, id2rgb, save_json

def show_semseg_result(result,msg_header, classes_id, classes_names):

    folder2 = "/home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg/src/panopticapi/"
    file_name2 = "panoptic_coco_categories.json"

    categories_json_file = os.path.join(folder2 + file_name2)

    with open(categories_json_file, 'r') as f:
        categories_list = json.load(f)
    categories = {el['id']: el for el in categories_list}



    id_generator = IdGenerator(categories)

    #pan_segm_id = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint32)
    pan_segm_id = np.zeros((480, 640), dtype=np.uint32)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # To Do: change logic of multiple annotations case, for now, it is override
    # but we can easily learn it from panoptic combine script
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    predicted_categories = list(np.unique(result))

    for category in predicted_categories:
        if category == 0.0: continue
        segment_id = id_generator.get_id(classes_id[int(category)])
        print("||||||||||id: " + str(classes_id[int(category)]))
        mask = (np.isin(result, category) * 1)

        pan_segm_id[mask==1] = segment_id
        print("||||||||||segment_id: " + str(segment_id))

        #print("shape: " id2rgb(pan_segm_id).shape)

    #print(sem_by_image)
    out_image_file = "pspnet_semseg_result_" + str(msg_header.stamp.to_sec()) + ".png"
    Image.fromarray(id2rgb(pan_segm_id)).save(
        os.path.join("/home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg/data/predictions/test_result_for_vox_ros/", out_image_file))



