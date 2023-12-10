import os
import cv2

# Define the paths to the data and output folders
data_folder = "Data"
output_folder = "ProcessedData"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define the target image size and file extensions to consider
target_size = (100, 100)
valid_extensions = [".jpg", ".jpeg", ".png"]

# Function to process and save an image
def process_and_save_image(image_path, output_path):
    # Read the image in grayscale mode
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image to the target size
    img = cv2.resize(img, target_size)

    # Save the preprocessed image
    cv2.imwrite(output_path, img)

# Recursively process images in subfolders of the "Data" folder
for root, dirs, files in os.walk(data_folder):
    for filename in files:
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            image_path = os.path.join(root, filename)

            # Construct the corresponding output path
            relative_path = os.path.relpath(image_path, data_folder)
            output_path = os.path.join(output_folder, relative_path)

            # Ensure the output folder structure exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Process and save the image
            process_and_save_image(image_path, output_path)

print("Data preprocessing complete.")
