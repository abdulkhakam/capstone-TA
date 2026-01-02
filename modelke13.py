import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Konfigurasi Path

DATA_DIR = r"C:/Users/LENOVO/OneDrive - mail.unnes.ac.id/kuliah/skripsi/waste-categorized-3"
ARTIFACTS_DIR = r"C:/Users/LENOVO/OneDrive - mail.unnes.ac.id/kuliah/skripsi"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

KERAS_PATH = os.path.join(ARTIFACTS_DIR, "wastecategorized13.keras")
H5_PATH    = os.path.join(ARTIFACTS_DIR, "wastecategorized13.h5")        
TFLITE_PATH= os.path.join(ARTIFACTS_DIR, "wastecategorized13.tflite")
CLASS_IDX_JSON = os.path.join(ARTIFACTS_DIR, "class_indices.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
SEED = 42

# ===============================
# Data Generator (train/valid)
# ===============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    # Augmentasi untuk bantu generalisasi
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="training",
    seed=SEED
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation",
    seed=SEED
)

# Simpan mapping kelas 
print("[INFO] class_indices:", train_gen.class_indices)
with open(CLASS_IDX_JSON, "w", encoding="utf-8") as f:
    json.dump(train_gen.class_indices, f, ensure_ascii=False, indent=2)


# Model: MobileNetV2 Transfer Learning

base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)
base_model.trainable = False  # tahap freeze backbone

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(train_gen.num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)



# Simpan model
# Format Keras
model.save(KERAS_PATH)
# Format H5 
model.save(H5_PATH)


# Konversi ke TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"[DONE] Disimpan ke:\n- {KERAS_PATH}\n- {H5_PATH}\n- {TFLITE_PATH}\n- {CLASS_IDX_JSON}")

