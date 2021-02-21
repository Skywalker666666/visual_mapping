#!/usr/bin/env /home/zhiliu/anaconda3/envs/panoptic_segmentation/bin/python

"""Semantic Segmentation performed by a PSPNet."""
import os
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

import mxnet as mx
import numpy as np
from mxnet import image
from tqdm import tqdm

from gluoncv.model_zoo.segbase import *
from gluoncv import model_zoo

# for DataParallelModel
from gluoncv.utils.parallel import *


#from src.models.pspnet import PSPNet
import sys
sys.path.insert(1,'./')
from psp_semseg_ros.predict_general import Predict



#default=len(mx.test_utils.list_gpus()),
## handle contexts
#if args.no_cuda:
#    print('Using CPU')
#    args.kvstore = 'local'
#    args.ctx = [mx.cpu(0)]
#else:
#    print('Number of GPUs:', args.ngpus)
#    args.ctx = [mx.gpu(i) for i in range(args.ngpus)]
#
#parser.add_argument('--resume', type=str, default=None,
#                    help='put the path to resuming file if needed')



class SemanticSegmentation(Predict):
    """Perform the semantic segmentation on a dataset by using a PSPNet."""

    #def __init__(self, model_path: str = '../../models/PSPNet_resnet50_1_epoch.params',
    #def __init__(self, model_path: str = '../../models/PSP_Resnet50_1epoch_auxon_batchs1_lr1en5.params',
    def __init__(self, model_path, no_cuda):

        print("no_cuda: " + str(no_cuda) )
        Predict.__init__(self, no_cuda)
        self.model_path = model_path
        self.model = self._load_model()
      
        if no_cuda:
            print('Using CPU')
            self.ctx = [mx.cpu(0)]
        else:
            ngpus=len(mx.test_utils.list_gpus())
            print('Number of GPUs:', ngpus)
            self.ctx = [mx.gpu(i) for i in range(ngpus)]
        print(self.ctx)

    @property
    def classes_name(self):
        return ['banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower',
                'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
                'railroad',
                'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick',
                'wall-stone',
                'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged',
                'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged',
                'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged',
                'food-other-merged',
                'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']
        #return ['thing', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower',
        #        'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
        #        'railroad',
        #        'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick',
        #        'wall-stone',
        #        'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged',
        #        'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged',
        #        'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged',
        #        'food-other-merged',
        #        'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']

    @property
    def classes(self):
        """Category coco index"""
        return [92, 93, 95, 100, 107, 109, 112, 118, 119, 122, 125, 128, 130, 133, 138, 141, 144, 145, 147, 148,
                149, 151, 154, 155, 156, 159, 161, 166, 168, 171, 175, 176, 177, 178, 180, 181, 184, 185, 186,
                187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200]
        #return [0, 92, 93, 95, 100, 107, 109, 112, 118, 119, 122, 125, 128, 130, 133, 138, 141, 144, 145, 147, 148,
        #        149, 151, 154, 155, 156, 159, 161, 166, 168, 171, 175, 176, 177, 178, 180, 181, 184, 185, 186,
        #        187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200]

    def _load_model(self):
        """Load PSPNet and the trained parameters.

        Returns
        -------

        The model with the associate parameters.

        """
        model = model_zoo.PSPNet(nclass=53, backbone='resnet50', pretrained_base=False, ctx=self.ctx)
        model.load_parameters(self.model_path)
        print("model_path: " + self.model_path)
        return model

    def predict(self, img):
        """Predict the semantic segmentation on a dataset.

        Parameters
        ----------
        img_path : str
            The directory path where the images are stored.

        Returns
        -------

        """
        model = self.model

        # The results will be stored in a list, it needs memory...
        # TODO: flush the memory periodically
        semantic_segmentation = []

        #print("image list: ")
        #print(self._imgs_idx)
        # self._imgs_idx demo: [466319, 523573, 308929, 57540]
        #tbar = tqdm(self._imgs_idx)

        #tbar = tqdm([466319, 523573])
        #tbar = tqdm([466319])

        #for idx in tbar:
        #img_metadata = coco.loadImgs(idx)[0]
        #path = img_metadata['file_name']
        #img_metadata = {'file_name': 'scenenn_camera_frame_76.550544_image.jpg'}
        #path = img_metadata['file_name']
        #img = image.imread(os.path.join(img_path, path))


        img = self.transform(img)


        #method 1:
        # comes with original reference code, demo()
        img = img.expand_dims(0).as_in_context(self.ctx[0])
        output = model.demo(img)
        predict_result = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

        #method 2: (prefered)
        # support multiple input image size
        ##img = img.expand_dims(0).as_in_context(self.ctx[0])
        ##evaluator = MultiEvalModel(model, 53, ctx_list=self.ctx)
        ##output = evaluator.parallel_forward(img)
        ##predict_result = mx.nd.squeeze(mx.nd.argmax(output[0], 1)).asnumpy()

 
        #method 3: # a quick version for no cuda
        # but this one is so slow
        # after read source code, we figure out this one
        #img2 = img.as_in_context(self.ctx[0])
        #print(self.ctx)
        #evaluator = MultiEvalModel(model, 53, ctx_list=self.ctx)
        #output = evaluator(img2)
        #predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()


        #method 4:
        # this method can only generate 480 x 480, cropped version
        #img = img.expand_dims(0).as_in_context(self.ctx[0])
        #evaluator = DataParallelModel(SegEvalModel(model), self.ctx)
        #outputs = evaluator(img.astype(self.dtype, copy=False))
        #output = [x[0] for x in outputs]
        #predict = mx.nd.squeeze(mx.nd.argmax(output[0], 1)).asnumpy()


        print('predict_result shape: ')
        print(predict_result.shape) 
        
        return predict_result

'''
        predicted_categories = list(np.unique(predict))

        
        #print(idx)
        #print(predicted_categories)

        for category in predicted_categories:
            print('Category: ')
            print(category)
            print("category_id: self.classes[int(category)]")
            print(self.classes[int(category)])
            print(self.classes_name[int(category)])
            # TODO: I think the category 0 is not 'banner' as expected... Need to look at the training.
            if category == 0.0: continue
            binary_mask = (np.isin(predict, category) * 1)
            binary_mask = np.asfortranarray(binary_mask).astype('uint8')
            segmentation_rle = coco_mask.encode(binary_mask)
            result = {"image_id": int(idx),
                      "category_id": self.classes[int(category)],
                      "segmentation": segmentation_rle,
                      }
            semantic_segmentation.append(result)
            #print('for category')
            #print(len(result['segmentation']))

        #print('for idx in tbar: result[segmentation]')
        #print(result['segmentation'])
        #print(len(semantic_segmentation))
        self.predictions = semantic_segmentation
'''

#if __name__ == "__main__":
#    args = parse_args()
#    predictor = SemanticSegmentation(args)
#    predictor.predict()
#    predictor.save_predictions(destination_dir='../../data/predictions', filename=args.sem_seg_result_file)
