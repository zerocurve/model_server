#
# Copyright (c) 2018-2019 Intel Corporation
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

from tensorflow_serving.apis import get_model_metadata_pb2
from tensorflow_serving.apis import get_model_status_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc, \
    model_service_pb2_grpc

from ie_serving.logger import get_logger
from ie_serving.server.constants import WRONG_MODEL_SPEC, \
    INVALID_METADATA_FIELD, SIGNATURE_NAME, GRPC
from ie_serving.server.get_model_metadata_utils import \
    prepare_get_metadata_output
from ie_serving.server.predict_utils import prepare_output_as_list, \
    prepare_input_data, StatusCode, statusCodes
from ie_serving.server.request import Request
from ie_serving.server.service_utils import \
    check_availability_of_requested_model, \
    check_availability_of_requested_status, add_status_to_response
import threading
import zmq
import SharedArray as sa
import numpy as np
import threading

logger = get_logger(__name__)


class PredictionServiceServicer(prediction_service_pb2_grpc.
                                PredictionServiceServicer):

    def __init__(self, process_id, in_queue, out_queue):
        self.process_id = process_id
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.zmq_context = zmq.Context()
        self.sockets = {}

        self.mocked_output = np.zeros((1, 1000), dtype="float32")

    def Predict(self, request, context):
        """
        Predict -- provides access to loaded TensorFlow model.
        """
        # check if requested model
        # is available on server with proper version
        thread_id = threading.get_ident()
        client_id = "{}:{}".format(self.process_id, thread_id)
        if thread_id not in self.sockets.keys():
            self.sockets[thread_id] = {
                'memory': self.zmq_context.socket(zmq.REQ),
                'inference_input': self.zmq_context.socket(zmq.REQ),
                'inference_output': self.zmq_context.socket(zmq.REP)
            }
            self.sockets[thread_id]['memory'].connect("ipc://memory.sock")
            self.sockets[thread_id]['inference_input'].connect("ipc://inference_in.sock")
            self.sockets[thread_id]['inference_output'].bind("ipc://{}.sock".format(client_id))
        inference_in_socket = self.sockets[thread_id]['inference_input']
        inference_out_socket = self.sockets[thread_id]['inference_output']
        memory_socket = self.sockets[thread_id]['memory']

        model_name = request.model_spec.name
        requested_version = request.model_spec.version.value

        deserialization_start_time = datetime.datetime.now()
        inference_input, error_message = \
            prepare_input_data(data=request.inputs,
                               service_type=GRPC)
        duration = (datetime.datetime.now() -
                    deserialization_start_time).total_seconds() * 1000
        if error_message is not None:
            code = statusCodes['invalid_arg'][GRPC]
            context.set_code(code)
            context.set_details(error_message)
            logger.debug("PREDICT, problem with input data. Exit code {}"
                         .format(code))
            return predict_pb2.PredictResponse()
        #logger.info("Sending io slot request")
        start_time = datetime.datetime.now()
        memory_socket.send_multipart([bytes(0), bytes(0), bytes(0)])
        #logger.info("Receiving io slot request")
        [in_bytes, out_bytes] = memory_socket.recv_multipart()
        duration = (datetime.datetime.now() -
                    start_time).total_seconds() * 1000
        logger.debug("Acquiring IO slot: {} ms".format(duration))
        #logger.info("Received io slot request")
        input_name = in_bytes.decode(encoding='ascii')
        output_name = out_bytes.decode(encoding='ascii')
        input_array = sa.attach(input_name)
        output_array = sa.attach(output_name)
        start_time = datetime.datetime.now()
        input_array[:] = inference_input['input']
        duration = (datetime.datetime.now() -
                    start_time).total_seconds() * 1000
        logger.debug("Copying input to shared memory: {} ms".format(duration))

        #logger.info("Sending inference signal")
        start_time = datetime.datetime.now()
        inference_in_socket.send_multipart([client_id.encode(encoding='ascii'), in_bytes, out_bytes])
        inference_in_socket.recv()
        duration = (datetime.datetime.now() -
                    start_time).total_seconds() * 1000
        logger.debug("Simple zmq send-recv: {} ms".format(duration))
        #logger.info(client_id)
        #logger.info("Receiving confirmation")
        inference_out_socket.recv()
        inference_out_socket.send(b'')
        #logger.info("Confirmation received")
        output = output_array
        inference_output = {"prob": output} #"resnet_v1_50/predictions/Softmax"
        serialization_start_time = datetime.datetime.now()
        response = prepare_output_as_list(
            inference_output=inference_output,
            model_available_outputs={"prob": "prob"}) # {"resnet_v1_50/predictions/Softmax": "resnet_v1_50/predictions/Softmax"})
        response.model_spec.name = model_name
        response.model_spec.version.value = 1
        response.model_spec.signature_name = SIGNATURE_NAME
        duration = (datetime.datetime.now() -
                    serialization_start_time).total_seconds() * 1000
        logger.debug("PREDICT; inference results serialization completed;"
                     " {}; {}; {} ms".format(model_name, 1, duration))
        #logger.info("Sending io slot free request")
        start_time = datetime.datetime.now()
        memory_socket.send_multipart([bytes(1), in_bytes, out_bytes])
        #logger.info("Receiving confirmation")
        memory_socket.recv_multipart()
        duration = (datetime.datetime.now() -
                    start_time).total_seconds() * 1000
        logger.debug("Memory freeing: {} ms".format(duration))
        return response

    def GetModelMetadata(self, request, context):

        # check if model with was requested
        # is available on server with proper version
        logger.debug("MODEL_METADATA, get request: {}".format(request))
        model_name = request.model_spec.name
        requested_version = request.model_spec.version.value
        valid_model_spec, version = check_availability_of_requested_model(
            models=self.models, requested_version=requested_version,
            model_name=model_name)

        if not valid_model_spec:
            context.set_code(StatusCode.NOT_FOUND)
            context.set_details(WRONG_MODEL_SPEC.format(model_name,
                                                        requested_version))
            logger.debug("MODEL_METADATA, invalid model spec from request")
            return get_model_metadata_pb2.GetModelMetadataResponse()
        target_engine = self.models[model_name].engines[version]
        metadata_signature_requested = request.metadata_field[0]
        if 'signature_def' != metadata_signature_requested:
            context.set_code(StatusCode.INVALID_ARGUMENT)
            context.set_details(INVALID_METADATA_FIELD.format
                                (metadata_signature_requested))
            logger.debug("MODEL_METADATA, invalid signature def")
            return get_model_metadata_pb2.GetModelMetadataResponse()

        inputs = target_engine.net.inputs
        outputs = target_engine.net.outputs

        signature_def = prepare_get_metadata_output(inputs=inputs,
                                                    outputs=outputs,
                                                    model_keys=target_engine.
                                                    model_keys)
        response = get_model_metadata_pb2.GetModelMetadataResponse()

        model_data_map = get_model_metadata_pb2.SignatureDefMap()
        model_data_map.signature_def['serving_default'].CopyFrom(
            signature_def)
        response.metadata['signature_def'].Pack(model_data_map)
        response.model_spec.name = model_name
        response.model_spec.version.value = version
        logger.debug("MODEL_METADATA created a response for {} - {}"
                     .format(model_name, version))
        return response


class ModelServiceServicer(model_service_pb2_grpc.ModelServiceServicer):

    def __init__(self, models):
        self.models = models

    def GetModelStatus(self, request, context):
        logger.debug("MODEL_STATUS, get request: {}".format(request))
        model_name = request.model_spec.name
        requested_version = request.model_spec.version.value
        valid_model_status = check_availability_of_requested_status(
            models=self.models, requested_version=requested_version,
            model_name=model_name)

        if not valid_model_status:
            context.set_code(StatusCode.NOT_FOUND)
            context.set_details(WRONG_MODEL_SPEC.format(model_name,
                                                        requested_version))
            logger.debug("MODEL_STATUS, invalid model spec from request")
            return get_model_status_pb2.GetModelStatusResponse()

        response = get_model_status_pb2.GetModelStatusResponse()
        if requested_version:
            version_status = self.models[model_name].versions_statuses[
                requested_version]
            add_status_to_response(version_status, response)
        else:
            for version_status in self.models[model_name].versions_statuses. \
                    values():
                add_status_to_response(version_status, response)

        logger.debug("MODEL_STATUS created a response for {} - {}"
                     .format(model_name, requested_version))
        return response
