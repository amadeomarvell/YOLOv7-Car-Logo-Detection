import os
import imghdr

test_images_folder = '/Users/amade/OneDrive/Desktop/SKRIPSI/YOLOv7-Car-Logo-Detection/data/Sample/'
wpodnet_model_path = '/Users/amade/OneDrive/Desktop/SKRIPSI/YOLOv7-Car-Logo-Detection/wpodnet.pth'
annotated_folder = '/Users/amade/OneDrive/Desktop/SKRIPSI/YOLOv7-Car-Logo-Detection/data/annotated_plates/'
warped_folder = '/Users/amade/OneDrive/Desktop/SKRIPSI/YOLOv7-Car-Logo-Detection/data/warped_plates/'
os.makedirs(annotated_folder, exist_ok=True)
os.makedirs(warped_folder, exist_ok=True)

for filename in os.listdir(test_images_folder):
    full_path = os.path.join(test_images_folder, filename)

    # Check if the item is a file
    if os.path.isfile(full_path):
        # Use imghdr to determine the image type
        image_type = imghdr.what(full_path)

        # Check if the image_type is not None (indicating it's a valid image)
        if image_type:
            input_image_path = full_path
            print(f'Processing: {input_image_path}')

            # Check if the file exists
            if os.path.isfile(input_image_path):
                command = f'python3 /Users/amade/OneDrive/Desktop/SKRIPSI/YOLOv7-Car-Logo-Detection/predict.py {input_image_path} -w {wpodnet_model_path} --save-annotated {annotated_folder} --save-warped {warped_folder}'
                print(f'Command: {command}')

                # Run the command
                os.system(command)
            else:
                print(f'Error: File not found - {input_image_path}')
        else:
            print(f'Skipping non-image file: {filename}')
