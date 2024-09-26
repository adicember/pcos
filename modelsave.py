from tensorflow.keras.models import load_model

def save_trained_model(model, save_path='saved_model/pcos_detection_model.h5'):
    """
    Saves the trained model to the specified path.

    Parameters:
    model (keras.Model): The trained model to be saved.
    save_path (str): The file path where the model will be saved.
    """
    model.save(save_path)
    print(f'Model saved to {save_path}')

# Example usage:
# Assuming 'model' is your trained model
# save_trained_model(model)
