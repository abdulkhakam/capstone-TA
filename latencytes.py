import time
import numpy as np
import tensorflow as tf

MODEL_PATH = "D:\skripsi\wastecategorized13.tflite"

print("Loading model...", flush=True)
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("INPUT DETAILS:", input_details, flush=True)
print("OUTPUT DETAILS:", output_details, flush=True)

input_shape = input_details[0]["shape"]
input_dtype = input_details[0]["dtype"]

print("Input shape:", input_shape, "dtype:", input_dtype, flush=True)

dummy = np.random.rand(*input_shape).astype(np.float32)

# Sesuaikan dtype 
if input_dtype != np.float32:
    dummy = (dummy * 255).astype(input_dtype)

# Warmup
print("Warmup...", flush=True)
for i in range(5):
    interpreter.set_tensor(input_details[0]["index"], dummy)
    interpreter.invoke()

print("Benchmark...", flush=True)
N = 30   
start = time.time()

try:
    for i in range(N):
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()
        if i % 10 == 0:
            print(f"Step {i}/{N}", flush=True)
except Exception as e:
    print("ERROR during invoke:", e, flush=True)
    raise

end = time.time()
avg_latency = (end - start) / N
fps = 1 / avg_latency

print(f"\nAverage Latency: {avg_latency*1000:.2f} ms", flush=True)
print(f"FPS: {fps:.2f}", flush=True)
