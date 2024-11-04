import os
import keras
import numpy as np
from keras.models import load_model

################# Load Model #################
def load_model_(model_name, summary=False, space='huggingface'):
    """
    Summerize: Load the model from '/Models' directory
    Args:
        - model_name: Name of the model to be loaded (without .keras)
        
    Issues:
        - Currently the function only support .keras model
    """
    if space == 'local':
        os.environ["KERAS_BACKEND"] = "tensorflow"
        # model_path = f"Models/{model_name}.keras"
        model_path = 'C:/Users/USER/Documents/Projects/Google_ML_BootCamp/NIPA/a-yo-image-model/a-yo-image-model_module/Models/unetv2_rgbmse.keras'
        print(f"Model Path: {model_path}")
        model = load_model(model_path)
    elif space == 'huggingface':
        os.environ["KERAS_BACKEND"] = "tensorflow"
        model = keras.saving.load_model("hf://mk48/nipa-cunet")

    if summary:
        """
        Summerize: Print the summary of the model
        Args:
            - model: Model to be summerized
        """
        model.summary()
    
    return model

################# Generate Image #################
def predict(model, x, label):
    """
    Summerize: Predict the image using the model
    Args:
        - model: Model to be used for prediction
        - dataset: Image to be predicted
    Input:
        - (new_x_test, test_label)
    Output:
        - ???
        
        
    Issues:
        - Currently the function only support .keras model (unetv2_rgbmse.keras)
    """
    
    new_x = x.repeat(10, axis=0)
    test_label = np.zeros((10, 10))

    for i in range(10) :
        test_label[i, i] = 1
        
    decoded_imgs = model.predict((new_x, test_label))
    decoded_imgs = decoded_imgs / 255
    print("Prediction Completed")
    
    return decoded_imgs