import numpy as np
import tensorflow as tf

MODEL_PATH = "D:/skripsi/wastecategorized13.tflite"
TEST_DIR = "D:/skripsi/test"
IMG_SIZE = (224,224)
BATCH_SIZE = 32

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names
C = len(class_names)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]["index"]
output_index = output_details[0]["index"]
input_dtype = input_details[0]["dtype"]

def preprocess(x):
    x = tf.cast(x, tf.float32) / 255.0
    x = x.numpy()
    if input_dtype != np.float32:
        x = (x * 255).astype(input_dtype)
    return x

conf_mat = np.zeros((C, C), dtype=np.int64)

for batch_x, batch_y in test_ds:
    batch_x = preprocess(batch_x)
    batch_y = batch_y.numpy()
    bs = batch_x.shape[0]

    interpreter.resize_tensor_input(input_index, [bs, IMG_SIZE[0], IMG_SIZE[1], 3])
    interpreter.allocate_tensors()

    interpreter.set_tensor(input_index, batch_x)
    interpreter.invoke()
    out = interpreter.get_tensor(output_index)
    preds = np.argmax(out, axis=1)

    for t, p in zip(batch_y, preds):
        conf_mat[t, p] += 1

# Metrics from confusion matrix
tp = np.diag(conf_mat)
fp = np.sum(conf_mat, axis=0) - tp
fn = np.sum(conf_mat, axis=1) - tp

precision = tp / np.maximum(tp + fp, 1)
recall    = tp / np.maximum(tp + fn, 1)
f1        = 2 * precision * recall / np.maximum(precision + recall, 1e-9)

accuracy = np.sum(tp) / np.sum(conf_mat)

print(f"\nOverall Accuracy: {accuracy*100:.2f}%\n")
print("Per-class metrics:")
for i, name in enumerate(class_names):
    print(f"{name:15s}  P={precision[i]:.3f}  R={recall[i]:.3f}  F1={f1[i]:.3f}")

print("\nConfusion Matrix:\n", conf_mat)
