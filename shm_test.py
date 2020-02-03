import multiprocessing
import numpy as np

def worker(arr, shape):
    np_arr = np.frombuffer(arr, dtype=np.float64).reshape(shape)
    np_arr[0][0] = [6, 6, 6]


X_shape = (3, 3, 3)
# Randomly generate some data
data = np.random.randn(*X_shape)
X = multiprocessing.RawArray('d', X_shape[0] * X_shape[1] * X_shape[2])
# Wrap X as an numpy array so we can easily manipulates its data.
X_np = np.frombuffer(X, dtype=np.float64).reshape(X_shape)
# Copy data to our shared array.
np.copyto(X_np, data)
print(X_np)
worker_process = multiprocessing.Process(target=worker, args=(X, X_shape))
worker_process.start()
worker_process.join()
print(X_np)
