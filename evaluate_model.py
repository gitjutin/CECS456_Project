import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 

from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre
from sklearn.metrics import classification_report, confusion_matrix

# --- Configuration & Paths ---
IMG_SHAPE = (224, 224)
BATCH_SZ = 32
SEED = 42

# Model file is in the same directory as this script
MODEL_PATH = 'natural_images_resnet_final.h5' 

# Dataset folder is in the same directory as this script
ROOT_DIR = 'natural_images' 

NUM_CLASSES = 8 

CLASS_NAMES = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']

# --- Step 1: Preprocessing Function ---
def preprocess(ds):
    def _resnet_preprocess(image, label):
        # Apply ResNet's specific preprocessing to raw image data
        return resnet_pre(image), label
    ds = ds.map(_resnet_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.cache().prefetch(tf.data.AUTOTUNE)


# --- Step 2: Load Dataset ---
print(f"Loading ALL dataset images from: {os.path.abspath(ROOT_DIR)}...")

full_ds = tf.keras.utils.image_dataset_from_directory(
    ROOT_DIR,
    image_size=IMG_SHAPE,
    label_mode='categorical',
    batch_size=BATCH_SZ,
    seed=SEED,
    shuffle=True
)

CLASS_NAMES = full_ds.class_names
NUM_CLASSES = len(CLASS_NAMES)

print("Detected class order:", CLASS_NAMES)

# Preprocess the entire dataset
preprocessed_full_ds = preprocess(full_ds)

# 1. Convert the entire dataset into NumPy arrays for accurate splitting
images = []
labels = []
for img_batch, label_batch in preprocessed_full_ds.unbatch().as_numpy_iterator():
    images.append(img_batch)
    labels.append(label_batch)

images = np.array(images)
labels = np.array(labels)

# 2. Determine the test split size (30% of the total dataset)
TOTAL_SIZE = len(images)
TEST_SIZE = int(TOTAL_SIZE * 0.30)
TRAIN_SIZE = TOTAL_SIZE - TEST_SIZE

# 3. Pull the final 30% of the data as the robust test set
test_images = images[TRAIN_SIZE:] 
test_labels = labels[TRAIN_SIZE:]

print(f"Extracted {len(test_images)} images for the Test Set (Approx. 30% of total).")

# 4. Re-batch the test images for prediction
test_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(BATCH_SZ)


# --- Step 3: Loading the Trained Model ---
print(f"\nLoading the trained model from: {os.path.abspath(MODEL_PATH)}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
    model.summary()
except Exception as e:
    print(f"\n=== ERROR ===")
    print(f"Error loading model: {e}")
    print("Please ensure the file 'natural_images_resnet_final.h5' is in the same directory.")
    exit()


# --- Step 4: Final Evaluation ---
print("\nEvaluating on test data...")

test_loss, test_acc = model.evaluate(test_data, verbose=1)
print(f"\nTest Accuracy: {test_acc*100:.2f}% | Test Loss: {test_loss:.4f}")

# Get predictions
y_true = np.concatenate([label for _, label in test_data], axis=0)
y_pred_probs = model.predict(test_data)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true_labels = np.argmax(y_true, axis=1)

print("\n=== Test Set Class Distribution ===")
unique, counts = np.unique(y_true_labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"{CLASS_NAMES[u]}: {c}")

# --- Step 5: Classification Report ---
print("\n=== Classification Report ===")
ALL_LABELS = np.arange(NUM_CLASSES)

print(classification_report(
    y_true_labels,
    y_pred,
    target_names=CLASS_NAMES,
    labels=ALL_LABELS
))


# --- Step 6: Confusion Matrix Visualization ---
conf_matrix = confusion_matrix(y_true_labels, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix (Final Test)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix_test_full.png')
plt.show()