import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "D:/skripsi/wastecategorized13.tflite"
TEST_DIR = "D:/skripsi/test"
IMG_SIZE = (224,224)
BATCH_SIZE = 32

# dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names
y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)

# tflite
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_dtype = input_details[0]["dtype"]

def preprocess(x):
    x = tf.cast(x, tf.float32) / 255.0
    x = x.numpy()
    if input_dtype != np.float32:
        x = (x * 255).astype(input_dtype)
    return x

y_pred = []
for batch_x, _ in test_ds:
    batch_x = preprocess(batch_x)
    for i in range(batch_x.shape[0]):
        inp = np.expand_dims(batch_x[i], axis=0)
        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]["index"])[0]
        y_pred.append(np.argmax(out))

y_pred = np.array(y_pred)

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_true, y_pred))
