from ultralytics import YOLO
import os
import shutil
import matplotlib.pyplot as plt

# Paths to dataset folders
data_path = "beverages_dataset"
new_data_path = "yolo_dataset"
train_path = os.path.join(data_path, "train")
val_path = os.path.join(data_path, "valid")
test_path = os.path.join(data_path, "test")
yaml_file = os.path.join(new_data_path, "yolo_dataset.yaml")

# Function to create YOLO-compatible dataset structure
def create_yolo_structure():
    """Create a new YOLO-compatible dataset structure with labels."""
    if os.path.exists(new_data_path):
        shutil.rmtree(new_data_path)  # Remove existing structure
    os.makedirs(new_data_path)

    for split in ["train", "valid", "test"]:
        original_path = os.path.join(data_path, split)
        new_images_path = os.path.join(new_data_path, split, "images")
        new_labels_path = os.path.join(new_data_path, split, "labels")
        os.makedirs(new_images_path, exist_ok=True)
        os.makedirs(new_labels_path, exist_ok=True)

        for class_name in os.listdir(original_path):
            class_path = os.path.join(original_path, class_name)
            if os.path.isdir(class_path):
                images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                for image in images:
                    # Copy image to the new images folder
                    src_image = os.path.join(class_path, image)
                    dst_image = os.path.join(new_images_path, image)
                    shutil.copy(src_image, dst_image)

                    # Create label file in the new labels folder
                    label_file = os.path.splitext(image)[0] + ".txt"
                    dst_label = os.path.join(new_labels_path, label_file)
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
def create_yaml_file():
    """Create a YOLO dataset YAML file."""
    os.makedirs(os.path.dirname(yaml_file), exist_ok=True)
    with open(os.path.join(data_path, "classes.txt"), "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    with open(yaml_file, "w") as yaml:
        yaml.write(f"train: {os.path.abspath(os.path.join(new_data_path, 'train', 'images'))}\n")
        yaml.write(f"val: {os.path.abspath(os.path.join(new_data_path, 'valid', 'images'))}\n")
        yaml.write(f"test: {os.path.abspath(os.path.join(new_data_path, 'test', 'images'))}\n")
        yaml.write("names:\n")
        for i, class_name in enumerate(class_names):
            yaml.write(f"  {i}: {class_name}\n")

# Training the YOLO model
def train_yolo():
    """Train YOLO on the dataset."""
    print("Starting YOLO training...")

    # Debug: Print YAML file content
    with open(yaml_file, "r") as f:
        print("YAML file content:")
        print(f.read())

    model = YOLO("yolov8n.pt")  # Use YOLOv8 nano pre-trained model
    results = model.train(
        data=yaml_file,  # Dataset YAML file
        epochs=20,       # Number of epochs
        imgsz=640,       # Image size
        batch=8,         # Batch size
        name="beverages_model",  # Model name
        patience=5,      # Early stopping patience
        verbose=True     # Verbosity
    )
    print("Training completed. Checkpoints saved in 'runs/train/beverages_model/'.")
    return model, results

# Validate the model
def validate_yolo(model):
    """Validate YOLO on the validation dataset."""
    print("Validating YOLO model...")
    metrics = model.val()
    print("Validation completed.")
    return metrics

# Test the model
def test_yolo(model):
    """Test YOLO on the test dataset."""
    print("Testing YOLO model...")
    results = model.predict(source=os.path.join(new_data_path, "test", "images"), save=True, save_txt=True, save_conf=True)
    print("Testing completed.")
    return results

# Function to plot training results
def plot_training_results(results):
    """Plot training results including loss and mAP."""
    metrics = results.metrics
    epochs = range(1, len(metrics["box_loss"]) + 1)

    plt.figure(figsize=(10, 6))

    # Loss Plot
    plt.subplot(2, 1, 1)
    plt.plot(epochs, metrics["box_loss"], label="Box Loss")
    plt.plot(epochs, metrics["cls_loss"], label="Class Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()

    # mAP Plot
    plt.subplot(2, 1, 2)
    plt.plot(epochs, metrics["mAP50"], label="mAP@50")
    plt.plot(epochs, metrics["mAP50_95"], label="mAP@50:95")
    plt.xlabel("Epochs")
    plt.ylabel("mAP")
    plt.title("Mean Average Precision Over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    create_yolo_structure()  # Create new dataset structure
    create_yaml_file()       # Create YOLO YAML file

    # Train YOLO
    print("Initializing training...")
    model = YOLO("yolov8n.pt")
    train_progress = model.train(
        data=yaml_file,
        epochs=20,
        imgsz=640,
        batch=8,
        name="beverages_model",
        patience=5,  # Early stopping after 5 epochs without improvement
        verbose=True
    )

    # Validate YOLO
    val_metrics = validate_yolo(model)

    # Test YOLO
    test_results = test_yolo(model)

    # Visualize Training Results
    plot_training_results(train_progress)
