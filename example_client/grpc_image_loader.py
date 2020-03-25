#
# Copyright (c) 2018-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np


def load_images(path):
    # optional preprocessing depending on the model
    images = np.load(path, mmap_mode='r', allow_pickle=False)
    images = images - np.min(images)  # Normalization 0-255
    images = images / np.ptp(images) * 255  # Normalization 0-255
    #images = images[:,:,:,::-1] # RGB to BGR
    print('Image data range:', np.amin(images), ':', np.amax(images))
    # optional preprocessing depending on the model


def load_labels(path):
    if path is not None:
        labels = np.load(path, mmap_mode='r', allow_pickle=False)
        return labels


def enlarge_to_batch_size(np_array, batch_size):
    while batch_size >= np_array.shape[0]:
        np_array = np.append(np_array, np_array, axis=0)
    return np_array


def transpose_input(images, method):
    if method == "nhwc2nchw":
        images = images.transpose((0, 3, 1, 2))
    if method == "nchw2nhwc":
        images = images.transpose((0, 2, 3, 1))
    return images
