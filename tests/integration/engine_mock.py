#
# Copyright (c) 2020 Intel Corporation
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

import threading
import zmq
import os
import numpy as np
import datetime
import queue
from concurrent.futures import ThreadPoolExecutor
from ie_serving.config import GLOBAL_CONFIG

import multiprocessing
from multiprocessing import shared_memory
from ie_serving.messaging.apis.endpoint_responses_pb2 \
    import EndpointResponse, PredictResponse
from ie_serving.messaging.apis.data_attributes_pb2 import NumpyAttributes
from ie_serving.messaging.apis.endpoint_requests_pb2 import EndpointRequest

outputs = {"output": np.zeros((1, 1000), dtype=np.float32)}
connections_map = {}
output_slots = {}
index_queue = queue.Queue()

def prepare_ipc_endpoint_response(ireq_index):
    ipc_endpoint_response = EndpointResponse()
    ipc_predict_response = PredictResponse()
    ipc_outputs = []

    for output_name in list(outputs.keys()):
        single_output = outputs[output_name]
        """
        start_time = datetime.datetime.now()
        output_shm = shared_memory.SharedMemory(create=True,
                                                size=single_output.nbytes)
        duration = (datetime.datetime.now() - start_time).total_seconds() * 1000
        print("Step 3 - Shared memory alloc for the results: {} ms".format(duration))

        start_time = datetime.datetime.now()
        shm_array = np.ndarray(single_output.shape, dtype=single_output.dtype,
                               buffer=output_shm.buf)
        duration = (datetime.datetime.now() - start_time).total_seconds() * 1000
        print("Step 4 - Make ndarray from shm buffer: {} ms".format(duration))
        """
        #start_time = datetime.datetime.now()
        (output_shm, shm_array) = output_slots[ireq_index]        
        #duration = (datetime.datetime.now() - start_time).total_seconds() * 1000
        #print("Step 3 and 4 - Prepare shm array: {} ms".format(duration))

        #start_time = datetime.datetime.now()
        shm_array[:] = single_output
        #duration = (datetime.datetime.now() - start_time).total_seconds() * 1000
        #print("Step 5 - Copy results to shared memory: {} ms".format(duration))

        #start_time = datetime.datetime.now()
        ipc_numpy_attributes = NumpyAttributes()
        ipc_numpy_attributes.shape.extend(list(shm_array.shape))
        ipc_numpy_attributes.data_type = shm_array.dtype.name

        ipc_output_data = PredictResponse.Data()
        ipc_output_data.numpy_attributes.CopyFrom(ipc_numpy_attributes)
        ipc_output_data.output_name = output_name
        ipc_output_data.shm_name = output_shm.name
        ipc_outputs.append(ipc_output_data)
        #duration = (datetime.datetime.now() - start_time).total_seconds() * 1000
        #print("Step 6 - Add output to IPC message: {} ms".format(duration))

    #start_time = datetime.datetime.now()
    ipc_predict_response.outputs.extend(ipc_outputs)
    ipc_predict_response.responding_version = 1
    ipc_endpoint_response.predict_response.CopyFrom(ipc_predict_response)
    #duration = (datetime.datetime.now() - start_time).total_seconds() * 1000
    #print("Step 7 - Form final IPC response message: {} ms".format(duration))
    return ipc_endpoint_response


def free_inputs_shm(ipc_predict_request):
    for ipc_input in ipc_predict_request.inputs:
        shm = shared_memory.SharedMemory(name=ipc_input.shm_name)
        shm.close()
        shm.unlink()

def send_response(ireq_index, return_socket_name):
    #start_time = datetime.datetime.now()
    thread_id = threading.get_ident()
    if thread_id not in connections_map:
        connections_map[thread_id] = {}
    if return_socket_name not in connections_map[thread_id]:        
        zmq_context = zmq.Context()
        return_socket = zmq_context.socket(zmq.REQ)
        return_socket.connect("ipc://{}".format(return_socket_name))
        connections_map[thread_id][return_socket_name] = return_socket
    return_socket = connections_map[thread_id][return_socket_name]
    #duration = (datetime.datetime.now() - start_time).total_seconds() * 1000
    #print("Step 2 - Select ZMQ socket to send IPC response: {} ms".format(duration))

    ipc_endpoint_response = prepare_ipc_endpoint_response(ireq_index)
    #start_time = datetime.datetime.now()
    msg = ipc_endpoint_response.SerializeToString()
    return_socket.send(msg)
    #duration = (datetime.datetime.now() - start_time).total_seconds() * 1000
    #print("Step 8 - Serialize and send IPC response message: {} ms".format(duration))
    return_socket.recv()
    index_queue.put(ireq_index)
    return null

def run_fake_engine():
    zmq_context = zmq.Context()
    engine_socket_name = os.path.join(GLOBAL_CONFIG['tmp_files_dir'],
                                      "{}-{}.sock".format("fake-model", 1))
    engine_socket = zmq_context.socket(zmq.REP)
    engine_socket.bind("ipc://{}".format(engine_socket_name))
    executor = ThreadPoolExecutor(max_workers=32)
    for i in range(32):
        index_queue.put(i)
        output_shm = shared_memory.SharedMemory(create=True,
                                                size=outputs['output'].nbytes)
        shm_array = np.ndarray(outputs['output'].shape, dtype=outputs['output'].dtype,
                               buffer=output_shm.buf)
        output_slots[i] = (output_shm, shm_array)

    print("Starting listening for inference requests")
    while True:
        ipc_raw_request = engine_socket.recv()
        #start_time = datetime.datetime.now()
        data = {}
        ipc_endpoint_request = EndpointRequest()
        ipc_endpoint_request.MergeFromString(ipc_raw_request)
        if not ipc_endpoint_request.HasField("predict_request"):
            continue
        for inference_input in ipc_endpoint_request.predict_request.inputs:
            shm = multiprocessing.shared_memory.SharedMemory(name=inference_input.shm_name)
            data[inference_input.input_name] = np.ndarray(
                tuple(inference_input.numpy_attributes.shape),
                dtype=inference_input.numpy_attributes.data_type, buffer=shm.buf)
        #duration = (datetime.datetime.now() - start_time).total_seconds() * 1000
        #print("Step 1 - IPC message deserialization: {} ms".format(duration))

        engine_socket.send(b'ACK')
        return_socket_name = ipc_endpoint_request.predict_request.\
            return_socket_name
        ireq_index = index_queue.get()
        executor.submit(send_response, ireq_index, return_socket_name) # simulating async inference scheduling

# This script imitates inference engine with a name "fake-model", version 1.
# It listens for requests from the server process and sends back a valid
# message. To run it with docker container, mount this script and run it
# alongside model server.


if __name__ == "__main__":
    run_fake_engine()
