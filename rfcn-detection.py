import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from time import sleep

import cv2

# cap = cv2.VideoCapture("//Users//bidushi//Downloads//gset-rfcn-export-data//training_video_smart_car.mp4")
cap = cv2.VideoCapture(1)


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("//Users//bidushi//Downloads//gset-rfcn-export-data//models//research")
sys.path.append("//Users//bidushi//Downloads//gset-rfcn-export-data//models//research//object_detection")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '//Users//bidushi//Downloads//gset-rfcn-export-data//frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '//Users//bidushi//Downloads//gset-rfcn-export-data//label_map.pbtxt'

NUM_CLASSES = 2

i = 1

# Load tensorflow model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
print(categories)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# PATH_TO_TEST_IMAGES_DIR = 'C:/Users/micha/OneDrive/Documents/GitHub/CVML-GSET-Project/dataset/test-images'
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'test{}.jpg'.format(i)) for i in range(1, 4) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

graph = detection_graph
with graph.as_default():
    with tf.Session() as sess:
        while True:
            ret, image = cap.read()
            image = cv2.resize(image, (480, 360))
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
            vis_util.visualize_boxes_and_labels_on_image_array(image, output_dict['detection_boxes'],
              output_dict['detection_classes'], output_dict['detection_scores'],
              category_index, instance_masks=output_dict.get('detection_masks'),
              use_normalized_coordinates=True, line_thickness=8)
            # cv2.rectangle(image, (180, 110), (300, 250), (0, 255, 0), 2)
            cv2.imshow('Object Detection', cv2.resize(image, (800, 600)))
            
            
            if(cv2.waitKey(25) & 0xFF == ord('q')):
                cv2.destroyAllWindows()
                break
            
            # analysis of intersection is triggered by space bar
            if cv2.waitKey(0) & 0xFF == ord(' '): 
                # cv2.imwrite("test_case_" + str(i) + ".jpg", image)
                
                e = 0
                boxes = [[], []]
                for box in output_dict['detection_boxes']:
                    if sum(box) / len(box) != 0.0:
                        if output_dict['detection_scores'][e] >= 0.5:
                            print(box)
                            print(output_dict['detection_scores'][e])
                            
                            # Self Driving Car is detected
                            if output_dict['detection_classes'][e] == 1:
                                boxes[0].append(box.tolist()) # index 0 for Self Driving Car
                            # Human Driven Car is detected
                            elif output_dict['detection_classes'][e] == 2:
                                boxes[1].append(box.tolist()) # index 1 for Human Driven Car
                            
                            e += 1
                            
                print(boxes)
                        
                print("Frame saved as test_case" + str(i))
                i += 1
