#!/usr/bin/env /home/zhiliu/anaconda3/envs/panoptic_segmentation/bin/python

import json

import mxnet as mx
from mxnet.gluon.data.vision import transforms

from pycocotools import mask
from pycocotools.coco import COCO


class Predict:
    def __init__(self, images_info_path: str = 'data/annotations/mini_test.json', no_cuda: bool = True):
        #self._coco = COCO(images_info_path)
        #self._coco_mask = mask
        self.ctx = self._define_context(no_cuda)
        #self._imgs_idx = self._imgs_idx()
        self.transform = self._make_transform()
        self.predictions = []

    @staticmethod
    def _define_context(no_cuda: bool = True) -> list:
        """Define the context of the computations (CPU or GPU).

        Parameters
        ----------
        no_cuda : bool
            If True, the computations will happen on the CPU.

        Returns
        -------
        A list of the available computation device.

        """
        if no_cuda:
            ctx = [mx.cpu(0)]
        else:
            ctx = [mx.gpu(0)]
        return ctx

#    def _imgs_idx(self):
#        return list(self._coco.imgs.keys())

    @staticmethod
    def _make_transform() -> mx.gluon.data.vision.transforms.Compose:
        """Transform the images.

        Returns
        -------
        Transformations applied on the images.

        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
            #transforms.Normalize([1, 1, 1], [1, 1, 1])
        ])

    def save_predictions(self, destination_dir: str, filename: str, empty_memory: bool = True):
        """Convert the binary strings in utf-8 encoding and dump the predictions in a json file.

        Parameters
        ----------
        destination_dir : str
            The directory where will be saved the predictions.

        filename : str
            The name of the file that contains the predictions.

        empty_memory : bool
            Since the predictions list take a lot of place in the memory, when the json file is written,
            one can use this tag to remove it from the memory.

        Returns
        -------

        """
        for prediction in self.predictions:
            prediction['segmentation']['counts'] = prediction['segmentation']['counts'].decode("utf-8")

        with open(destination_dir + '/' + filename, 'w') as f:
            json.dump(self.predictions, f)
        print('The predictions were just saved in: {}'.format(destination_dir + '/' + filename))

        if empty_memory:
            self.predictions = []
