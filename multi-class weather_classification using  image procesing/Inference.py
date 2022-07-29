import tensorflow as tf
import sys
import os
import cv2
import random
import numpy as np
from PIL import Image

vidcap = cv2.VideoCapture('video.mp4')
success,image = vidcap.read()
count = 0
image_lst = []
while success:
    success,image = vidcap.read()
    image_lst.append(image)
    # print('Read a new frame: ', success)
    if count == 50:
        break
    count += 1

check_lst = random.sample(image_lst, 10)

# speicherorte fuer trainierten graph und labels in train.sh festlegen ##

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

	# holt labels aus file in array
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("D:/Capstone_project1/capstone_DL/Multi-class Weather Dataset/model/output_labels.txt")]
	# !! labels befinden sich jeweils in eigenen lines -> keine aenderung in retrain.py noetig -> falsche darstellung im windows editor !!

with tf.gfile.FastGFile("D:/Capstone_project1/capstone_DL/Multi-class Weather Dataset/model/output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()	# The graph-graph_def is a saved copy of a TensorFlow graph; objektinitialisierung
    graph_def.ParseFromString(f.read())	#Parse serialized protocol buffer data into variable
    _ = tf.import_graph_def(graph_def, name='')

# img_path = "D:/Capstone_project1/capstone_DL/Multi-class Weather Dataset/model/New_folder"
# angabe in console als argument nach dem aufruf
with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    output_dict = {}
    for image in check_lst:
        image_data = cv2.imencode('.jpg', image)[1].tostring()

        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        for node_id in top_k:
            if node_id == 0:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                if human_string not in output_dict.keys():
                    output_dict[human_string] = 1
                else:
                    output_dict[human_string] += 1

final_prediction = max(zip(output_dict.values(), output_dict.keys()))[1]
print("\n \n Today, it looks like weather is: ",final_prediction)
