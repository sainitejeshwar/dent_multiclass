#LIBRARIES
import os
import cv2
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import skimage
import glob
import itertools
from scipy.misc import imsave, imread, imresize

#MRCNN Model
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

#PYTHON FILES
import colorsys
import train_multiclass

#FLASK 
from flask import render_template
from flask import Flask
from flask import request
from flask import jsonify
from flask import send_file

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR)

#For importing the model
def load_presaved_model(WEIGHTS_PATH):
    config = train_multiclass.CustomConfig()
    custom_DIR = os.path.join(ROOT_DIR, "dataset")
    class InferenceConfig(config.__class__):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    DEVICE = "/cpu:0"
    TEST_MODE = "inference"
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                  config=config)
    print("Loading weights:", WEIGHTS_PATH)
    model.load_weights(WEIGHTS_PATH, by_name=True)
    print("---Model Loaded---")
    return model

WEIGHTS_PATH = "mask_rcnn_part_0100.h5"
model = load_presaved_model(WEIGHTS_PATH)

#Utils
def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors
def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

#Saving results
def result_save(image_number , image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """

    classes = ['dent','scratch']
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    colors = colors or random_colors(N)

    height, width = image.shape[:2]
   
    masked_image = image.copy()
    for i in range(N):
        color = colors[i]

        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            color_temp = list(color)
            for cl in range(len(color_temp)):
                color_temp[cl]  = 255*color_temp[cl]
            color_temp = tuple(color_temp)
            cv2.rectangle(image , (x1, y1), (x2 , y2), color_temp,5)

        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = classes[class_ids[i]-1]
            caption = label  
        else:
            caption = captions[i]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image,caption,(x1,y1+10), font, 1,(255,255,255),2,cv2.LINE_AA)
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(image, mask, color)
    result_name = 'result.jpg'
    cv2.imwrite(result_name,masked_image)
    print("Result saved as : " , result_name)



#Flask Main code

graph = tf.get_default_graph()
app = Flask(__name__)
@app.route('/predict',methods=['POST'])
def predict():
    with graph.as_default():
        str1=request.files['file']
        print(str1)
        image = imread(str1)
        results = model.detect([image], verbose=1)
        print("Detection Done..")
        r = results[0]
        image_number = 1000
        result_save(image_number ,image, r['rois'], r['masks'], r['class_ids'], 
                            'damage', r['scores'], 
                            title="Predictions")
        return send_file("result.jpg", mimetype='image/gif')



if(__name__=='__main__'):
    app.run()






