import numpy as np
import tensorflow as tf
from data_preprocessing import img_height, img_width, is_mask_class

def predict_color(model, img_path, class_names):
    """
    Predict the color of an image.
    
    Args:
        model (tf.keras.Model): The trained model.
        img_path (str): Path to the image file.
        class_names (list): List of class names.
    """
    img = tf.keras.utils.load_img(
        img_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    if is_mask_class(predicted_class):
        color = predicted_class.split('_')[1]
        print(f"This image most likely contains a {color} mask with a {confidence:.2f} percent confidence.")
    else:
        print(f"This image most likely contains the color {predicted_class} with a {confidence:.2f} percent confidence.")
    
    # Check for corresponding mask/non-mask class
    check_corresponding_class(predicted_class, score, class_names)

def check_corresponding_class(predicted_class, score, class_names):
    """
    Check if there's a high probability for the corresponding mask/non-mask class.
    
    Args:
        predicted_class (str): The predicted class name.
        score (tf.Tensor): The prediction scores.
        class_names (list): List of class names.
    """
    class_to_index = {name: index for index, name in enumerate(class_names)}
    color_classes = [c for c in class_names if not c.startswith('mask_')]
    
    if predicted_class in color_classes:
        mask_class = f"mask_{predicted_class}"
        mask_score = score[class_to_index[mask_class]]
        if mask_score > 0.3:
            print(f"There's also a {mask_score:.2f} probability that this is a masked version of {predicted_class}.")
    elif predicted_class.startswith('mask_'):
        color_class = predicted_class.split('_')[1]
        color_score = score[class_to_index[color_class]]
        if color_score > 0.3:
            print(f"There's also a {color_score:.2f} probability that this is an unmasked version of {color_class}.")

# data_preprocessing.py
# (Add this function to the existing data_preprocessing.py file)

def is_mask_class(class_name):
    """Determine if a class is a mask class."""
    return class_name.startswith('mask_')