import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Paths
base_dir = r"C:\Users\Goutham C S\Desktop\model2"
dataset_dir = os.path.join(base_dir, "PlantVillage")
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "val")  # optional if already exists
split_dir = os.path.join(base_dir, "PlantVillage_split")

# Create split dirs if not exist
for folder in ["train", "validation", "test"]:
    os.makedirs(os.path.join(split_dir, folder), exist_ok=True)

# Copy training images
for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    train_class_dir = os.path.join(split_dir, "train", class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    for img in os.listdir(class_path):
        shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))

# Split validation set into validation + test
def split_validation():
    for class_name in os.listdir(val_dir):
        class_path = os.path.join(val_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        images = os.listdir(class_path)
        val_imgs, test_imgs = train_test_split(images, test_size=0.5, random_state=42)

        val_class_dir = os.path.join(split_dir, "validation", class_name)
        test_class_dir = os.path.join(split_dir, "test", class_name)
        os.makedirs(val_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        for img in val_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(val_class_dir, img))
        for img in test_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(test_class_dir, img))

split_validation()

# Data generators
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(split_dir, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)
validation_generator = val_datagen.flow_from_directory(
    os.path.join(split_dir, "validation"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)
test_generator = val_datagen.flow_from_directory(
    os.path.join(split_dir, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Build model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Train
EPOCHS = 15
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# Save
model_path = os.path.join(base_dir, "plantvillage_multiclass_model.h5")
model.save(model_path)
print(f"✅ Model saved at {model_path}")

# Evaluate
test_loss, test_acc = model.evaluate(test_generator)
print(f"✅ Test Accuracy: {test_acc:.4f}")
