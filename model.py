from data_preprocessing import img_height, img_width
import tensorflow as tf

def create_model(num_classes):
    """
    Create the CNN model for color classification.
    
    Args:
        num_classes (int): Number of classes to predict.
    
    Returns:
        tf.keras.Model: The compiled model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(img_height, img_width, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])
    
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_ds, val_ds, epochs=10):
    """
    Train the model.
    
    Args:
        model (tf.keras.Model): The model to train.
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.
        epochs (int): Number of epochs to train for.
    
    Returns:
        tf.keras.Model: The trained model.
    """
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    
    return model

def train_model(model, train_ds, val_ds, epochs=10):
    """
    Train the model.
    
    Args:
        model (tf.keras.Model): The model to train.
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.
        epochs (int): Number of epochs to train for.
    
    Returns:
        tf.keras.Model: The trained model.
    """
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    
    return model