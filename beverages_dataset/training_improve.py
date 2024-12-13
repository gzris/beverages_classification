from ultralytics import YOLO
import os
import shutil
import matplotlib.pyplot as plt

# Paths to dataset folders
data_path = "beverages_dataset"
improved_data_path = "improved_yolo_dataset"
yaml_file = os.path.join(improved_data_path, "improved_yolo_dataset.yaml")

# Function to create an improved YOLO dataset structure
def create_improved_yolo_structure():
    """Create an improved YOLO-compatible dataset structure with labels."""
    if os.path.exists(improved_data_path):
        shutil.rmtree(improved_data_path)  # Remove existing structure
    os.makedirs(improved_data_path)

    for split in ["train", "valid", "test"]:
        original_path = os.path.join(data_path, split)
        images_path = os.path.join(improved_data_path, split, "images")
        labels_path = os.path.join(improved_data_path, split, "labels")
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
def create_improved_yaml_file():
    """Create a YOLO dataset YAML file."""
    os.makedirs(os.path.dirname(yaml_file), exist_ok=True)
    with open(os.path.join(data_path, "classes.txt"), "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    with open(yaml_file, "w") as yaml:
        yaml.write(f"train: {os.path.abspath(os.path.join(improved_data_path, 'train', 'images'))}\n")
        yaml.write(f"val: {os.path.abspath(os.path.join(improved_data_path, 'valid', 'images'))}\n")
        yaml.write(f"test: {os.path.abspath(os.path.join(improved_data_path, 'test', 'images'))}\n")
        yaml.write("names:\n")
        for i, class_name in enumerate(class_names):
            yaml.write(f"  {i}: {class_name}\n")

# Function to improve YOLO training with optimized settings
def train_improved_yolo():
    """Train YOLO with optimized settings for better performance."""
    print("Starting improved YOLO training...")

    model = YOLO("yolov8s.pt")  # Use YOLOv8 small pre-trained model for better performance

    # Optimized training settings
    results = model.train(
        data=yaml_file,  # Improved dataset YAML file
        epochs=25,       # Increased number of epochs for better convergence
        imgsz=512,       # Reduced image size for faster training
        batch=16,        # Increased batch size for stability
        optimizer="AdamW",  # Use AdamW optimizer for better optimization
        name="improved_beverages_model",  # Model name
        patience=3,      # Early stopping patience
        lr0=1e-3,        # Initial learning rate
        verbose=True,    # Verbosity
        augment=True     # Enable data augmentation
    )
    print("Improved training completed. Checkpoints saved in 'runs/train/improved_beverages_model/'.")
    return model, results

# Function to visualize improved training results
def visualize_training_results(results):
    """Visualize training results including loss and mAP metrics."""
    print("Generating training results plots...")
    results.plot()  # Automatically generates and saves plots

# Main execution
if __name__ == "__main__":
    create_improved_yolo_structure()  # Create improved dataset structure
    create_improved_yaml_file()      # Create improved YAML file

    # Train YOLO with improved settings
    print("Initializing improved training...")
    model, train_results = train_improved_yolo()

    # Visualize training results
    visualize_training_results(train_results)
