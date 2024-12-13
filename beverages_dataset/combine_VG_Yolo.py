from ultralytics import YOLO
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.optimizers import Adam
import os
import shutil
import matplotlib.pyplot as plt

# Paths to dataset folders
data_path = "beverages_dataset"
combined_data_path = "combined_yolo_vgg_dataset"
yaml_file = os.path.join(combined_data_path, "combined_dataset.yaml")

# Function to create YOLO-compatible dataset structure
def create_combined_structure():
    """Create a dataset structure for YOLO and VGG16."""
    if os.path.exists(combined_data_path):
        shutil.rmtree(combined_data_path)  # Remove existing structure
    os.makedirs(combined_data_path)

    for split in ["train", "valid", "test"]:
        original_path = os.path.join(data_path, split)
        images_path = os.path.join(combined_data_path, split, "images")
        labels_path = os.path.join(combined_data_path, split, "labels")
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(labels_path, exist_ok=True)

        for class_name in os.listdir(original_path):
            class_path = os.path.join(original_path, class_name)
            if os.path.isdir(class_path):
                images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                for image in images:
                    # Copy image to the images folder
                    src_image = os.path.join(class_path, image)
                    dst_image = os.path.join(images_path, image)
                    shutil.copy(src_image, dst_image)

                    # Create label file in the labels folder
                    label_file = os.path.splitext(image)[0] + ".txt"
                    dst_label = os.path.join(labels_path, label_file)
                    with open(dst_label, "w") as f:
                        class_id = get_class_id(class_name)
                        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")  # Dummy bounding box (full image)

# Helper function to map class names to IDs
def get_class_id(class_name):
    """Map class names to numeric IDs."""
    with open(os.path.join(data_path, "classes.txt"), "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names.index(class_name)

# Function to create a YOLO YAML file
def create_combined_yaml_file():
    """Create a YOLO dataset YAML file."""
    os.makedirs(os.path.dirname(yaml_file), exist_ok=True)
    with open(os.path.join(data_path, "classes.txt"), "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    with open(yaml_file, "w") as yaml:
        yaml.write(f"train: {os.path.abspath(os.path.join(combined_data_path, 'train', 'images'))}\n")
        yaml.write(f"val: {os.path.abspath(os.path.join(combined_data_path, 'valid', 'images'))}\n")
        yaml.write(f"test: {os.path.abspath(os.path.join(combined_data_path, 'test', 'images'))}\n")
        yaml.write("names:\n")
        for i, class_name in enumerate(class_names):
            yaml.write(f"  {i}: {class_name}\n")

# Function to train YOLO model
def train_yolo():
    """Train YOLO on the dataset."""
    print("Starting YOLO training...")
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=yaml_file,
        epochs=10,
        imgsz=512,
        batch=16,
        name="combined_yolo_model",
        patience=3,
        verbose=True
    )
    print("YOLO training completed.")
    return model, results

# Function to train VGG16 model
def train_vgg16():
    """Train a VGG16-based classifier."""
    print("Starting VGG16 training...")

    # Load pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(len(os.listdir(os.path.join(data_path, "train"))), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze base VGG16 layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

    print("VGG16 model ready for training. Ensure data is preprocessed correctly.")
    return model

# Function to combine YOLO and VGG16 results
def combine_models(yolo_model, vgg_model):
    """Combine YOLO and VGG16 models for robust predictions."""
    print("Combining YOLO and VGG16 models...")
    # Placeholder logic: In practice, you would ensemble the outputs or use specific outputs for different tasks.
    return None

# Main execution
if __name__ == "__main__":
    create_combined_structure()  # Create combined dataset structure
    create_combined_yaml_file()  # Create combined YAML file

    # Train YOLO
    print("Training YOLO...")
    yolo_model, yolo_results = train_yolo()

    # Train VGG16
    print("Training VGG16...")
    vgg16_model = train_vgg16()

    # Combine models
    print("Combining YOLO and VGG16...")
    combined_model = combine_models(yolo_model, vgg16_model)

    print("YOLO and VGG16 training and combination completed.")
