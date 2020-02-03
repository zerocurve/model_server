import grpc
import numpy as np
import tensorflow as tf
import time
import threading
import datetime
import queue

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

futures_queue = queue.Queue()


def main():
    channel = grpc.insecure_channel('localhost:9000')

    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()

    request.model_spec.name = 'resnet'

    input_data = np.ones((1, 3, 224, 224), dtype=np.float32)

    request.inputs['input'].CopyFrom(tf.make_tensor_proto(input_data))

    n = 10000
    workers = 8
    start_time = datetime.datetime.now()
    for i in range(n // workers):
        futures = [stub.Predict.future(request, 10.0) for i in range(workers)]
        for future in futures:
            future.result()
    duration = (datetime.datetime.now() -
                start_time).total_seconds() * 1000
    print("Async client duration for {} iterations using {} async calls at a time: {} s".format(n, workers, duration/1000))
    print("Average latency per request: {} ms".format(duration / n))

    start_time = datetime.datetime.now()
    for i in range(n):
        futures = stub.Predict(request, 10.0)
    duration = (datetime.datetime.now() -
                start_time).total_seconds() * 1000
    print("Sync client duration for {} iterations: {} s".format(n, duration/1000))
    print("Average latency per request: {} ms".format(duration/n))


if __name__ == '__main__':
    main()
