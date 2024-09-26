import cv2
import numpy as np
from tensorflow.keras.models import load_model
#inference code: to test if bahira ko image is recognised
def preprocess_image(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    # Resize the image to match the input shape required by the model (224x224)
    img = cv2.resize(img, (224, 224))
    
    # Normalize pixel values to the range [0, 1]
    img = img / 255.0
    
    # Expand dimensions to match the input shape (1, 224, 224, 1)
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    
    return img

def load_trained_model(model_path):
    # Load the pre-trained model
    model = load_model(model_path)
    return model

def predict_pcos(image_path, model_path):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    
    # Load the trained model
    model = load_trained_model(model_path)
    
    # Make prediction
    prediction = model.predict(preprocessed_image)[0][0]
    
    # Interpret the prediction and calculate confidence
    confidence_score = prediction * 100  # Convert to percentage
    
    if prediction > 0.5:
        result = "PCOS Detected"
    else:
        result = "No PCOS Detected"
        confidence_score = (1 - prediction) * 100  # Adjust confidence for 'No PCOS'
    
    return result, confidence_score

if __name__ == "__main__":
    image_path = '/Users/aditikasingh/pcos_detection/infectedImage.jpg'  # Specify the path to the image
    model_path = '/Users/aditikasingh/pcos_detection/saved_model/pcos_detection_model.h5'  # Specify the path to the trained model

    result, confidence = predict_pcos(image_path, model_path)
    print(f"Result: {result} with a confidence score of {confidence:.2f}%")
