import tensorflow as tf
import numpy as np

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# --- Basic Configuration ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
SEED = 42

DATA_DIR = "natural_images"
MODEL_NAME = "natural_images_resnet_final.h5"

tf.random.set_seed(SEED)
np.random.seed(SEED)

# --- Load Dataset ---
print("Loading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=0.2,
    subset="training",
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=0.2,
    subset="validation",
    label_mode="categorical"
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Class order:", class_names)

# --- Preprocessing --- 

def preprocess_data(ds):
    return ds.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).cache().prefetch(tf.data.AUTOTUNE)

train_ds = preprocess_data(train_ds)
val_ds = preprocess_data(val_ds)

# --- Build Model ---
print("Building model...")

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=IMG_SIZE + (3,)
)

# Freeze pretrained layers
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --- Train Model ---
print("Training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)


# --- Save Model ---
model.save(MODEL_NAME)
print(f"\nFinished! Model saved as {MODEL_NAME}")
