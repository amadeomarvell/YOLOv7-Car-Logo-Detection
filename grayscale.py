# pre-process test images using adaptive equalization for grayscale images
import cv2
import os

# Specify the paths to the input (original) and output (enhanced) folders
input_folder = '/Users/amade/OneDrive/Desktop/SKRIPSI/YOLOv7-Car-Logo-Detection/data/enhanced_images_3/'
output_folder = '/Users/amade/OneDrive/Desktop/SKRIPSI/YOLOv7-Car-Logo-Detection/data/grayscale_images_3/'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Function to apply adaptive histogram equalization to a grayscale image
def apply_adaptive_equalization_gray(image_path, output_path):
    # Read the grayscale image
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply adaptive histogram equalization to the grayscale image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_gray)

    # Save the enhanced image
    cv2.imwrite(output_path, img_enhanced)

# Iterate through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Full path to the input and output images
        input_image_path = os.path.join(input_folder, filename)
        output_image_path = os.path.join(output_folder, f'{filename}')

        # Apply adaptive histogram equalization to grayscale image
        apply_adaptive_equalization_gray(input_image_path, output_image_path)

# Display or further process the enhanced images as needed
