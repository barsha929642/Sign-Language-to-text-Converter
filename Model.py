import os
import numpy as np

from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the folder where your image data is stored
data_folder = "ProcessedData"

image_size = (100, 100)  # Input image size
num_classes = 26  # Number of classes (letters A to Z)
batch_size = 32
epochs = 10  # Adjust the number of training epochs as needed


# Create lists to store image data and labels
X = []
y = []

# Create a dictionary to map alphabet letters to integer labels
label_mapping = {'A': 0, 'B': 1, 'C': 2 ,'D': 3, 'E': 4, 'F': 5,'G': 6, 'H': 7, 'I': 8,'J': 9, 'K': 10, 'L': 11,'M': 12, 'N': 13, 'O': 14,'P': 15, 'Q': 16, 'R': 17,'S': 18, 'T': 19, 'U': 20,'V': 21, 'W': 22, 'X': 23,'Y': 24, 'Z': 25}  # Add more letters as needed

# Load and preprocess the image data
for root, dirs, files in os.walk(data_folder):
    for filename in files:
        if filename.endswith(".jpg"):
            # Load the image using Pillow (PIL)
            img = Image.open(os.path.join(root, filename))

            # Resize the image to the desired size
            img = img.resize(image_size)

            # Convert the image to a NumPy array
            img_array = np.array(img)

            # Get the label from the folder name
            folder_name = os.path.basename(root)
            label = label_mapping.get(folder_name, -1)  # Use -1 for unknown labels

            if label != -1:
                # Append the image data and label
                X.append(img_array)
                y.append(label)

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




model = keras.Sequential([
    keras.layers.Input(shape=image_size + (1,)),  # Input layer with the specified image size
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(26, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
# Specify the directory where you want to save the model
model_directory = "SavedModel"  # Replace with your desired path

# Save the trained model to the specified directory
model.save(os.path.join(model_directory, "keras_model.h5"))

# Specify the directory where you want to save the labels file
labels_directory = "SavedModel"  # Replace with your desired path

# Define your labels as a list
labels = ["A", "B", "C",'D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']  # Modify this list based on your actual labels

# Save the labels to a text file in the specified directory
labels_file_path = os.path.join(labels_directory, "labels.txt")
with open(labels_file_path, "w") as file:
    for label in labels:
        file.write(label + "\n")