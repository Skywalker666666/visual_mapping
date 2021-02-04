#!/usr/bin/env /home/zhiliu/anaconda3/envs/pspnet_kerastf/bin/python3
from __future__ import print_function
import os
from os.path import join, isfile

import sys
#ros_path = '/opt/ros/melodic/lib/python2.7/dist-packages'
ros_path2 = '/home/zhiliu/Documents/catkin_ws_VoSM/devel/lib/python2.7/dist-packages'
ros_path3 = '/home/zhiliu/Documents/Weapons/catkin_workspace_bridge/install/lib/python3.7/site-packages'
if ros_path2 in sys.path:
    sys.path.remove(ros_path2)

sys.path.insert(1,ros_path3)


import numpy as np
from scipy import misc, ndimage
from keras import backend as K
from keras.models import model_from_json, load_model
import tensorflow as tf

import layers_builder_2 as layers
from keras.utils.generic_utils import CustomObjectScope
import cv2
import math
# -- Fix for macos, uncomment it
# import matplotlib
# matplotlib.use('TkAgg')
# --
import matplotlib.pyplot as plt
from PIL import Image

from imageio import imread

sys.path.append(ros_path2)



# These are the means for the ImageNet pretrained ResNet
DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]])  # RGB order


class PSPNet(object):
    """Pyramid Scene Parsing Network by Hengshuang Zhao et al 2017"""

    def __init__(self, nb_classes, resnet_layers, input_shape, weights):
        self.input_shape = input_shape
        self.num_classes = nb_classes

        json_path = join("/home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg_PSP2/PSPNet-Keras-tensorflow/weights/weights", "keras", weights + ".json")
        h5_path = join("/home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg_PSP2/PSPNet-Keras-tensorflow/weights/weights", "keras", weights + ".h5")
        if 'pspnet' in weights:
            if os.path.isfile(json_path) and os.path.isfile(h5_path):
                print("Keras model & weights found, loading...")
                with CustomObjectScope({'Interp': layers.Interp}):
                    with open(json_path) as file_handle:
                        self.model = model_from_json(file_handle.read())
                self.model.load_weights(h5_path)
            else:
                print("No Keras model & weights found, import from npy weights.")
                self.model = layers.build_pspnet(nb_classes=nb_classes,
                                                 resnet_layers=resnet_layers,
                                                 input_shape=self.input_shape)
                self.set_npy_weights(weights)
        else:
            print('Load pre-trained weights')
            self.model = load_model(weights)

    def predict(self, img, flip_evaluation=False):
        """
        Predict segementation for an image.
        Arguments:
            img: must be rowsxcolsx3
        """

        if img.shape[0:2] != self.input_shape:
            print(
                "Input %s not fitting for network size %s, resizing. You may want to try sliding prediction for better results." % (
                img.shape[0:2], self.input_shape))

            # method 1
            img = misc.imresize(img, self.input_shape)

            # method 2
            #img = np.array(Image.fromarray(img.astype(np.uint8)).resize((self.input_shape[1], self.input_shape[0]), resample=Image.BILINEAR))


        img = img - DATA_MEAN
        img = img[:, :, ::-1]  # RGB => BGR
        img = img.astype('float32')

        probs = self.feed_forward(img, flip_evaluation)

        return probs

    def predict_sliding(self, full_img, flip_evaluation):
        """
        Predict on tiles of exactly the network input shape.
        This way nothing gets squeezed.
        """
        tile_size = self.input_shape
        classes = self.num_classes
        overlap = 1 / 3

        stride = math.ceil(tile_size[0] * (1 - overlap))
        tile_rows = max(int(math.ceil((full_img.shape[0] - tile_size[0]) / stride) + 1), 1)  # strided convolution formula
        tile_cols = max(int(math.ceil((full_img.shape[1] - tile_size[1]) / stride) + 1), 1)
        print("Need %i x %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, stride))
        full_probs = np.zeros((full_img.shape[0], full_img.shape[1], classes))
        count_predictions = np.zeros((full_img.shape[0], full_img.shape[1], classes))
        tile_counter = 0
        for row in range(tile_rows):
            for col in range(tile_cols):
                x1 = int(col * stride)
                y1 = int(row * stride)
                x2 = min(x1 + tile_size[1], full_img.shape[1])
                y2 = min(y1 + tile_size[0], full_img.shape[0])
                x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
                y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

                img = full_img[y1:y2, x1:x2]
                padded_img = self.pad_image(img, tile_size)
                plt.imshow(padded_img)
                plt.show()
                tile_counter += 1
                print("Predicting tile %i" % tile_counter)
                padded_prediction = self.predict(padded_img, flip_evaluation)
                prediction = padded_prediction[0:img.shape[0], 0:img.shape[1], :]
                count_predictions[y1:y2, x1:x2] += 1
                full_probs[y1:y2, x1:x2] += prediction  # accumulate the predictions also in the overlapping regions

        # average the predictions in the overlapping regions
        full_probs /= count_predictions
        # visualize normalization Weights
        # plt.imshow(np.mean(count_predictions, axis=2))
        # plt.show()
        return full_probs

    @staticmethod
    def pad_image(img, target_size):
        """Pad an image up to the target size."""
        rows_missing = target_size[0] - img.shape[0]
        cols_missing = target_size[1] - img.shape[1]
        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, 0)), 'constant')
        return padded_img


    def predict_multi_scale(self, img, flip_evaluation, sliding_evaluation, scales):
        """Predict an image by looking at it with different scales."""

        full_probs = np.zeros((img.shape[0], img.shape[1], self.num_classes))
        h_ori, w_ori = img.shape[:2]

        print("Started prediction...")
        for scale in scales:
            print("Predicting image scaled by %f" % scale)
            # method1
            # skywalker: misc.imresize is deprecated. so need find abetter way if we want it works in new scipy >=1.3.0
            # scale is a floating point number
            scaled_img = misc.imresize(img, size=scale, interp="bilinear")

            # method 2
            #scaled_img = np.array(Image.fromarray(img.astype(np.uint8)).resize((int(np.floor(img.shape[1]*scale)), int(np.floor(img.shape[0]*scale))), resample=Image.BILINEAR))


            # I have verified, the result of two methods are same

            if sliding_evaluation:
                scaled_probs = self.predict_sliding(scaled_img, flip_evaluation)
            else:
                scaled_probs = self.predict(scaled_img, flip_evaluation)

            # scale probs up to full size
            # visualize_prediction(probs)
            probs = cv2.resize(scaled_probs, (w_ori, h_ori))
            full_probs += probs
        full_probs /= len(scales)
        print("Finished prediction...")

        return full_probs

    def feed_forward(self, data, flip_evaluation=False):
        assert data.shape == (self.input_shape[0], self.input_shape[1], 3)

        if flip_evaluation:
            print("Predict flipped")
            input_with_flipped = np.array(
                [data, np.flip(data, axis=1)])
            prediction_with_flipped = self.model.predict(input_with_flipped)
            prediction = (prediction_with_flipped[
                          0] + np.fliplr(prediction_with_flipped[1])) / 2.0
        else:
            prediction = self.model.predict(np.expand_dims(data, 0))[0]
        return prediction

    def set_npy_weights(self, weights_path):
        npy_weights_path = join("weights", "npy", weights_path + ".npy")
        json_path = join("weights", "keras", weights_path + ".json")
        h5_path = join("weights", "keras", weights_path + ".h5")

        print("Importing weights from %s" % npy_weights_path)
        weights = np.load(npy_weights_path, encoding='bytes').item()
        for layer in self.model.layers:
            print(layer.name)
            if layer.name[:4] == 'conv' and layer.name[-2:] == 'bn':
                mean = weights[layer.name.encode()][
                    'mean'.encode()].reshape(-1)
                variance = weights[layer.name.encode()][
                    'variance'.encode()].reshape(-1)
                scale = weights[layer.name.encode()][
                    'scale'.encode()].reshape(-1)
                offset = weights[layer.name.encode()][
                    'offset'.encode()].reshape(-1)

                self.model.get_layer(layer.name).set_weights(
                    [scale, offset, mean, variance])

            elif layer.name[:4] == 'conv' and not layer.name[-4:] == 'relu':
                try:
                    weight = weights[layer.name.encode()]['weights'.encode()]
                    self.model.get_layer(layer.name).set_weights([weight])
                except Exception as err:
                    biases = weights[layer.name.encode()]['biases'.encode()]
                    self.model.get_layer(layer.name).set_weights([weight,
                                                                  biases])
        print('Finished importing weights.')

        print("Writing keras model & weights")
        json_string = self.model.to_json()
        with open(json_path, 'w') as file_handle:
            file_handle.write(json_string)
        self.model.save_weights(h5_path)
        print("Finished writing Keras model & weights")


class PSPNet50(PSPNet):
    """Build a PSPNet based on a 50-Layer ResNet."""

    def __init__(self, nb_classes, weights, input_shape):
        PSPNet.__init__(self, nb_classes=nb_classes, resnet_layers=50,
                        input_shape=input_shape, weights=weights)

class PSPNet101(PSPNet):
    """Build a PSPNet based on a 101-Layer ResNet."""

    def __init__(self, nb_classes, weights, input_shape):
        PSPNet.__init__(self, nb_classes=nb_classes, resnet_layers=101,
                        input_shape=input_shape, weights=weights)













