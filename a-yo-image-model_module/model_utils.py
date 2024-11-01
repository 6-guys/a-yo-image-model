from keras.models import load_model

################# Load Model #################
def load_model(model_name):
    """
    Summerize: Load the model from '/Models' directory
    Args:
        - model_name: Name of the model to be loaded (without .keras)
        
    Issues:
        - Currently the function only support .keras model
    """
    
    model = load_model(f"Models/{model_name}.keras")
    return model


def model_summary(model):
    """
    Summerize: Print the summary of the model
    Args:
        - model: Model to be summerized
    """
    model.summary()


################# Generate Image #################
def predict(model, dataset):
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
    
    predict = model.predict(dataset)
    
    return predict

