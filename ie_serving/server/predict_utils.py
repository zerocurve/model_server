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

import falcon
import numpy as np
from grpc import StatusCode
from tensorflow.core.framework import tensor_pb2
from tensorflow.python.framework import tensor_shape
from tensorflow_serving.apis import predict_pb2
from tensorflow.python.framework import dtypes as dtypes
from tensorflow.python.framework import tensor_util as tensor_util
import tensorflow.contrib.util as tf_contrib_util
# import tensorflow.contrib.util as tf_contrib_util
from ie_serving.models.shape_management.utils import BatchingMode, ShapeMode
from ie_serving.server.constants import \
    INVALID_INPUT_KEY, INVALID_SHAPE, INVALID_BATCHSIZE, GRPC, REST
from ie_serving.logger import get_logger

logger = get_logger(__name__)

statusCodes = {
    'invalid_arg': {GRPC: StatusCode.INVALID_ARGUMENT,
                    REST: falcon.HTTP_BAD_REQUEST},
}


def prepare_input_data(data, service_type):
    # returns:
    # inference_input, None on success
    # None, error_message on error
    model_inputs_in_input_request = list(dict(data).keys())
    inference_input = {}

    for requested_input_blob in model_inputs_in_input_request:

        tensor_name = requested_input_blob
        if service_type == GRPC:
            try:
                tensor_input = tf_contrib_util. \
                    make_ndarray(data[requested_input_blob])
            except Exception as e:
                message = str(e)
                logger.debug("PREDICT prepare_input_data make_ndarray error: "
                             "{}".format(message))
                return None, message
        else:
            tensor_input = np.asarray(data[requested_input_blob])

        inference_input[tensor_name] = tensor_input
    return inference_input, None


def prepare_output_as_list(inference_output, model_available_outputs):
    response = predict_pb2.PredictResponse()
    for key, value in model_available_outputs.items():
        if value in inference_output:
            dtype = dtypes.as_dtype(inference_output[value].dtype)
            output_tensor = tensor_pb2.TensorProto(
                dtype=dtype.as_datatype_enum,
                tensor_shape=tensor_shape.as_shape(
                    inference_output[value].shape).as_proto())
            result = inference_output[value].flatten()
            tensor_util._NP_TO_APPEND_FN[dtype.as_numpy_dtype](output_tensor,
                                                               result)
            response.outputs[key].CopyFrom(output_tensor)
    return response


'''
The function is not used.
Probably preparing the output would be faster,
but you need a change of grpc clients.

def prepare_output_with_tf(inference_output, model_available_outputs):
    response = predict_pb2.PredictResponse()

    for output in model_available_outputs:
        response.outputs[output].CopyFrom(
            tf_contrib_util.make_tensor_proto(inference_output[output],
                                              shape=inference_output[output].
                                              shape,
                                              dtype=dtypes.as_dtype(
                                                  inference_output
                                                  [output].dtype).
                                              as_datatype_enum))
    return response
'''
