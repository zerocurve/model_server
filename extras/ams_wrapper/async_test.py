from openvino.inference_engine import IENetwork, IEPlugin, IECore
import numpy as np
import os
import queue
import sys

import tensorflow as tf
from src.preprocessing import preprocess_binary_image

from matplotlib import pyplot as plt

model_base_path = "/tmp/ovms_models/saved_models/vehicle-detection-adas-binary-0001/1/vehicle-detection-adas-binary-0001"
model_xml = model_base_path + ".xml"
model_bin = model_base_path + ".bin"

IMAGES_DIR = os.path.join(os.path.dirname('/tmp/ovms_models/saved_data/vehicle-detection-adas-binary-0001/1/data/images/'))
# Plugin initialization for specified device and load extensions library if specified

core = IECore()

def png_image():
    #with open(os.path.join(IMAGES_DIR,'005405_001.png'), mode='rb') as img_file:
    #with open(os.path.join(IMAGES_DIR,'025448_001.png'), mode='rb') as img_file:
    with open('/home/rasapala/ams/vehicle.png', mode='rb') as img_file:
        binary_image = img_file.read()
    return binary_image


image = png_image()

array_img = preprocess_binary_image(image, channels = 3,
                            target_size = (384, 672),
                            channels_first = True,
                            dtype = tf.dtypes.uint8,
                            standardization = False,
                            reverse_input_channels = True)


#ndarray = np.full((300,300,3), 125, dtype=np.uint8)

#plt.imshow(array_img, interpolation='nearest')
#plt.show()

# Read IR
print("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
net = IENetwork(model=model_xml, weights=model_bin)
exec_net = core.load_network(network=net, device_name='CPU', num_requests=2)

#imgs = np.zeros((1,3,384,672))

print('Image data range:', np.amin(array_img), ':', np.amax(array_img))
print('Images in shape: {}\n'.format(array_img.shape))

img_list = []
img_list.append(array_img)

img_data = np.array(img_list)


print('Array data range:', np.amin(img_data), ':', np.amax(img_data))
print('Array in shape: {}\n'.format(img_data.shape))

input_blob = next(iter(net.inputs)) # to ci wyciaga nazwe inputu
out_blob = next(iter(net.outputs))

inference_output = exec_net.infer({input_blob: img_data})

print("Inference output: " + str(inference_output[out_blob]))



