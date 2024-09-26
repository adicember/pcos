import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from pcos_detection.data_loader import load_images
from pcos_detection.model import build_and_train_model
from pcos_detection.utils import plot_results

def main():
    # Load data
    X_train, y_train = load_images('image/train')
    X_test, y_test = load_images('image/test')
    
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Number of samples in train set: {len(X_train)}")

    # Standardize features and add channel dimension
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 64*64)).reshape(X_train.shape + (1,))
    X_test_scaled = scaler.transform(X_test.reshape(-1, 64*64)).reshape(X_test.shape + (1,))

    print(f"Shape of X_train_scaled: {X_train_scaled.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Number of samples in train set: {len(X_train)}")

    # Split train set into training and validation
    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

    # Build and train model
    model, history = build_and_train_model(X_train_final, y_train_final, X_val, y_val)

    # Save the trained model
    model.save('saved_model/pcos_detection_model.h5')
    print('Model saved to saved_model/pcos_detection_model.h5')

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
    print(f'Test loss: {test_loss:.4f}')
    print(f'Test accuracy: {test_accuracy:.2f}')

    # Plot results
    plot_results(history, model, X_test_scaled, y_test)

if __name__ == "__main__":
    main()
