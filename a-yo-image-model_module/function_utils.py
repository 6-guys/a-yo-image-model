from keras.saving import register_keras_serializable
from tensorflow.keras import backend as K

@register_keras_serializable()
def mse_rgb(y_true, y_pred):
    """
    Summarize: This function is a custom loss function that applies MSE specifically to RGB channels.
    Input: 
        - Shape: ((H, W, C), (H, W, C))
        - Type: ndarray
        - Example: ((128, 128, 4), (128, 128, 4))
    Output:
        - Shape: (1, )
        - Type: ndarray
        - Example: (64, 9, 128, 128, 4)
    """
    return K.mean(K.square(y_true - y_pred), axis=-1)