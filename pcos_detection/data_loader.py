import os
import cv2
import numpy as np

def load_images(directory):
    print(f"Checking directory: {directory}")
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return None, None
    
    X = []
    y = []
    for folder in ['infected', 'non_infected']:
        folder_path = os.path.join(directory, folder)
        print(f"Checking subfolder: {folder_path}")
        if not os.path.exists(folder_path):
            print(f"Subfolder does not exist: {folder_path}")
            continue
        
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            print(f"Checking image file: {img_path}")
            
            # Load the image in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue

            # Apply Gaussian blur for noise removal
            img = cv2.GaussianBlur(img, (5, 5), 0)  # Adjust kernel size as needed
            
            # Resize the image
            img = cv2.resize(img, (64, 64))

            # Normalize pixel values
            img_array = img / 255.0
            
            # Add to X and y
            X.append(img_array)
            y.append(1 if folder == 'infected' else 0)

    return np.array(X), np.array(y)
