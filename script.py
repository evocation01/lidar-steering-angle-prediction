import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

# Initialize TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Load the serialized TensorRT engine
with open("final_lidar_model.trt", "rb") as f:
    engine_data = f.read()

# Create a TensorRT runtime object
runtime = trt.Runtime(TRT_LOGGER)

# Deserialize the engine from the file
engine = runtime.deserialize_cuda_engine(engine_data)

# Create an execution context for the engine
context = engine.create_execution_context()

# Get input and output shapes
input_shape = engine.get_binding_shape(0)
output_shape = engine.get_binding_shape(1)

# Allocate memory for input and output buffers
input_buffer = cuda.mem_alloc(np.prod(input_shape) * np.float32().nbytes)
output_buffer = cuda.mem_alloc(np.prod(output_shape) * np.float32().nbytes)

# Prepare input data (example input)
input_host = np.random.randn(*input_shape).astype(np.float32)
output_host = np.empty(output_shape, dtype=np.float32)

# Transfer input data to device
cuda.memcpy_htod(input_buffer, input_host)

# Run inference
context.execute_v2([input_buffer, output_buffer])

# Transfer output data back to host
cuda.memcpy_dtoh(output_host, output_buffer)

# Print output
print(output_host)
