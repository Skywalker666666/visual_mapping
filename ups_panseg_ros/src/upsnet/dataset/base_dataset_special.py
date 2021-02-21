import numpy as np
from upsnet.config.config import config
import cv2

# *****************************************************************************************************
# function borrowed from upsnet/dataset/base_data.py
def prep_im_for_blob(im, pixel_means, target_sizes, max_size):
    """Prepare an image for use as a network input blob. Specially:
      - Subtract per-channel pixel mean
      - Convert to float32
      - Rescale to each of the specified target size (capped at max_size)
    Returns a list of transformed images, one for each target size. Also returns
    the scale factors that were used to compute each returned image.
    """
    im = im.astype(np.float32, copy=False)
    if config.network.use_caffe_model:
        im -= pixel_means.reshape((1, 1, -1))
    else:
        im /= 255.0
        im -= np.array([[[0.485, 0.456, 0.406]]])
        im /= np.array([[[0.229, 0.224, 0.225]]])
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    print("im_shape: ")
    print(im_shape)
    print(im_shape[0:2])

    print("im_size_min: ")
    print(im_size_min)
    print("im_size_max: ")
    print(im_size_max)

    ims = []
    im_scales = []
    for target_size in target_sizes:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than max_size
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        print("im_scale: ")
        print(im_scale)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        ims.append(im)
        im_scales.append(im_scale)
    return ims, im_scales



# function borrowed from upsnet/dataset/base_data.py
def im_list_to_blob(ims, scale=1):
    """Convert a list of images into a network input. Assumes images were
    prepared using prep_im_for_blob or equivalent: i.e.
      - BGR channel order
      - pixel means subtracted
      - resized to the desired input size
      - float32 numpy ndarray format
    Output is a 4D HCHW tensor of the images concatenated along axis 0 with
    shape.
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    # Pad the image so they can be divisible by a stride
    print("config.network.has_fpn: ")
    print(config.network.has_fpn)
    if config.network.has_fpn:
        stride = float(config.network.rpn_feat_stride[-2])
        max_shape[1] = int(np.ceil(max_shape[1] / stride) * stride)
        max_shape[2] = int(np.ceil(max_shape[2] / stride) * stride)

    num_images = len(ims)
    blob = np.zeros((num_images, 3, int(max_shape[1] * scale), int(max_shape[2] * scale)),
                    dtype=np.float32)
    for i in range(num_images):
        # transpose back to normal first, then after interpolate, transpose it back.
        im = ims[i] if scale == 1 else cv2.resize(ims[i].transpose(1, 2, 0), None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        blob[i, :, 0:im.shape[1], 0:im.shape[2]] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    return blob

def get_unified_pan_result(segs, pans, cls_inds, stuff_area_limit=4 * 64 * 64):
    pred_pans_2ch = []

    for (seg, pan, cls_ind) in zip(segs, pans, cls_inds):
        pan_seg = pan.copy()
        pan_ins = pan.copy()
        id_last_stuff = config.dataset.num_seg_classes - config.dataset.num_classes
        ids = np.unique(pan)
        ids_ins = ids[ids > id_last_stuff]
        pan_ins[pan_ins <= id_last_stuff] = 0
        for idx, id in enumerate(ids_ins):
            region = (pan_ins == id)
            if id == 255:
                pan_seg[region] = 255
                pan_ins[region] = 0
                continue
            cls, cnt = np.unique(seg[region], return_counts=True)
            if cls[np.argmax(cnt)] == cls_ind[id - id_last_stuff - 1] + id_last_stuff:
                pan_seg[region] = cls_ind[id - id_last_stuff - 1] + id_last_stuff
                pan_ins[region] = idx + 1
            else:
                if np.max(cnt) / np.sum(cnt) >= 0.5 and cls[np.argmax(cnt)] <= id_last_stuff:
                    pan_seg[region] = cls[np.argmax(cnt)]
                    pan_ins[region] = 0
                else:
                    pan_seg[region] = cls_ind[id - id_last_stuff - 1] + id_last_stuff
                    pan_ins[region] = idx + 1

        idx_sem = np.unique(pan_seg)
        for i in range(idx_sem.shape[0]):
            if idx_sem[i] <= id_last_stuff:
                area = pan_seg == idx_sem[i]
                if (area).sum() < stuff_area_limit:
                    pan_seg[area] = 255

        print("pan.shape: ")
        print(pan.shape)
        pan_2ch = np.zeros((pan.shape[0], pan.shape[1], 3), dtype=np.uint8)

        pan_2ch[:, :, 0] = pan_seg
        pan_2ch[:, :, 1] = pan_ins
        pred_pans_2ch.append(pan_2ch)
    return pred_pans_2ch



