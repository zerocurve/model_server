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


import datetime
import time
import grpc
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import multiprocessing
import threading

from grpc_args import parse_args
from grpc_image_loader import load_images, load_labels, transpose_input, enlarge_to_batch_size

args = parse_args()

batch_size = int(args.get('batchsize'))
images_path = args['images_numpy_path']
labels_path = args.get('labels_numpy_path')
imgs = load_labels(args['images_numpy_path'])
if labels_path is not None:
    lbs = load_labels(path=labels_path)
    lbs = enlarge_to_batch_size(np_array=lbs, batch_size=batch_size)
    matched_count = 0
    total_executed = 0

imgs = enlarge_to_batch_size(np_array=imgs, batch_size=batch_size)
if args.get('transpose_input') == "True":
    imgs = transpose_input(images=imgs, method=args.get('transpose_method'))

img = imgs[0:(0 + batch_size)]

iterations = int((imgs.shape[0]//batch_size) if not (args.get('iterations') or args.get('iterations') != 0) else args.get('iterations'))
threads = args.get('threads')
model_name = args.get('model_name')


def thread_function(worker, results, port, iterations, model_name):
    channel = grpc.insecure_channel("{}:{}".format(args['grpc_address'], args['grpc_port']))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    print("Created stub")
    total_time = 0
    thread_start_time = datetime.datetime.now()
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.inputs[args['input_name']].CopyFrom(tf.make_tensor_proto(img, shape=img.shape))
    for i in range(iterations):
        start_time = datetime.datetime.now()
        result = stub.Predict(request, 10.0) # result includes a dictionary with all model outputs
        end_time = datetime.datetime.now()
        request_time = (end_time - start_time).total_seconds() * 1000
        total_time = total_time + request_time

    thread_end_time = datetime.datetime.now()
    total_thread_time = (thread_end_time - thread_start_time).total_seconds() * 1000
    average_thread_time = total_thread_time/iterations
    average_time = total_time/iterations
    requests_per_second = 1000/average_time
    results[0] = total_thread_time
    results[1] = average_thread_time
    results[2] = average_time
    results[3] = requests_per_second


def main():
    print("Sending requests on multiple threads")
    port = 9005
    results = []
    for i in range(threads):
        results.append(multiprocessing.Array('d', range(4)))
    jobs = []

    for i in range(threads):
        job_handle = multiprocessing.Process(target=thread_function, args=(i, results[i], port, iterations, model_name))
        jobs.append(job_handle)
        job_handle.start()

    for job in jobs:
        job.join()

    total_thread_time = 0
    average_thread_time = 0
    average_time = 0
    requests_per_second = 0

    for result in results:
        total_thread_time = total_thread_time + result[0]
        average_thread_time = average_thread_time + result[1]
        average_time = average_time + result[2]
        requests_per_second = requests_per_second + result[3]

    total_thread_time = total_thread_time/threads
    average_thread_time = average_thread_time/threads
    average_time = average_time/threads

    #for i in range(threads):
    #    thread_handle = threading.Thread(target=thread_function, args=(port, iterations. model_name))
    #    thread_handle.start()

    print("Total requests: {}".format(iterations*threads))

    print("Total requests per second: {}".format(requests_per_second))
    print("Thread finished sending requests. Average time: {} ms".format(average_time))

    print("Average total thread time: {} ms".format(total_thread_time))
    print("Average thread time per image: {} ms".format(average_thread_time))
    print("Experiment overhead: {} ms".format(average_thread_time-average_time))


main()
