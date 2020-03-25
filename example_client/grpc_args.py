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

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Sends requests via TFS gRPC API using images in numpy format. '
                                                     'It displays performance statistics and optionally the model accuracy')
    parser.add_argument('--images_numpy_path', required=True, help='numpy in shape [n,w,h,c] or [n,c,h,w]')
    parser.add_argument('--labels_numpy_path', required=False, help='numpy in shape [n,1] - can be used to check model accuracy')
    parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
    parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service. default: 9000')
    parser.add_argument('--input_name',required=False, default='input', help='Specify input tensor name. default: input')
    parser.add_argument('--output_name',required=False, default='resnet_v1_50/predictions/Reshape_1',
                        help='Specify output name. default: resnet_v1_50/predictions/Reshape_1')
    parser.add_argument('--transpose_input', choices=["False", "True"], default="True",
                        help='Set to False to skip NHWC>NCHW or NCHW>NHWC input transposing. default: True',
                        dest="transpose_input")
    parser.add_argument('--transpose_method', choices=["nchw2nhwc","nhwc2nchw"], default="nhwc2nchw",
                        help="How the input transposition should be executed: nhwc2nchw or nchw2nhwc",
                        dest="transpose_method")
    parser.add_argument('--iterations', default=0,
                        help='Number of requests iterations, as default use number of images in numpy memmap. default: 0 (consume all frames)',
                        dest='iterations', type=int)
    # If input numpy file has too few frames according to the value of iterations and the batch size, it will be
    # duplicated to match requested number of frames
    parser.add_argument('--batchsize', default=1,
                        help='Number of images in a single request. default: 1',
                        dest='batchsize')
    parser.add_argument('--model_name', default='resnet', help='Define model name, must be same as is in service. default: resnet',
                        dest='model_name')
    parser.add_argument('--threads', default=1,
                        help='Number of threads sendig requests , default: 1',
                        dest='threads', type=int)
    args = vars(parser.parse_args())
    return args

