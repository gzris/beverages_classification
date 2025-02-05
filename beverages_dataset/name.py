import os

# Automatically detect the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
train_folder = os.path.join(current_dir, "train")
classes_file = os.path.join(current_dir, "classes.txt")

def collect_and_save_class_names(train_path, output_file):
    """
    Collects class names from subfolders in the train directory and saves them to a file.

    Args:
        train_path (str): Path to the train folder containing class subfolders.
        output_file (str): Path to the output file (classes.txt).
    """
    # Check if the train folder exists
    if not os.path.exists(train_path):
        print(f"Error: Train folder '{train_path}' does not exist.")
        return

    # Collect class names (subfolder names)
    class_names = [
        folder for folder in os.listdir(train_path)
        if os.path.isdir(os.path.join(train_path, folder))
    ]

    # Save the class names to the output file
    with open(output_file, "w") as file:
        for class_name in sorted(class_names):  # Sort alphabetically for consistency
            file.write(class_name + "\n")

    print(f"Class names saved to '{output_file}'.")

# Run the function
collect_and_save_class_names(train_folder, classes_file)

""" train: C:\Users\GzrGN\PycharmProjects\beverages\beverages_dataset\train
val: C:\Users\GzrGN\PycharmProjects\beverages\beverages_dataset\valid
test: C:\Users\GzrGN\PycharmProjects\beverages\beverages_dataset\test
names:
  0: 7UP 320ML
  1: UP 390ML
  2: 7UP free 320ML
  3: Aquafina Soda
  4: Lipton 455ML
  5: Mirinda Blueberry 320ML
  6: Mirinda Blueberry 390ML
  7: Mirinda Green Cream 320ML
  8: Mirinda Green Cream 390ML
  9: Mirinda Orange 320ML
  10: Mirinda Orange 390ML
  11: Mirinda Sarsi 320ML
  12: Mirinda Sarsi 390ML
  13: OPP
  14: Pepsi 320ML
  15: Pepsi 390ML
  16: Pepsi zero calo 320ML
  17: Pepsi zero calo 390ML
  18: Revive Isotonic Original
  19: Revive Salted Lemon
  20: Rockstar 250ML
  21: Sting 320ML
  22: Sting 330ML
  23: Sting Gold 320ML
  24: Sting Gold 330ML
  25: Tea Olong 320ML
  26: Tea Olong 450ML
  27: Tea Olong No Sugar 450ML
  28: Tea Plus Lemon 450ML
  29: Twister 455ML
  30: Twister Orange 350ML"""