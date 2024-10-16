import tensorflow as tf
import pathlib

# Constants
data_dir = pathlib.Path("ColorClassification")
batch_size = 32
img_height = 180
img_width = 180

def load_datasets():
    """
    Load and split the dataset into training and validation sets.
    
    Returns:
        tuple: (train_ds, val_ds, class_names)
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    class_names = train_ds.class_names
    return train_ds, val_ds, class_names

def color_jitter(image, jitter=0.2):
    """Apply random color jittering to the image."""
    image = tf.image.random_brightness(image, max_delta=jitter)
    image = tf.image.random_contrast(image, lower=1-jitter, upper=1+jitter)
    image = tf.image.random_saturation(image, lower=1-jitter, upper=1+jitter)
    image = tf.image.random_hue(image, max_delta=jitter)
    return image

def is_mask_class(class_name):
    """Determine if a class is a mask class."""
    return tf.strings.regex_full_match(class_name, "mask_.*")

def preprocess(image, label, class_names):
    """Preprocess the image and label."""
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    
    # Apply data augmentation
    image = data_augmentation(image)
    
    # Get the class name for the label
    class_name = tf.gather(class_names, label)
    
    # Apply color jittering only to non-mask images
    image = tf.cond(
        is_mask_class(class_name),
        lambda: image,
        lambda: color_jitter(image)
    )
    
    # Normalize the image
    image = normalization_layer(image)
    
    return image, label

def prepare_datasets():
    """
    Prepare the datasets for training.
    
    Returns:
        tuple: (train_ds, val_ds, class_names)
    """
    train_ds, val_ds, class_names = load_datasets()
    
    def prepare_dataset(ds):
        return ds.map(
            lambda x, y: preprocess(x, y, tf.constant(class_names)), 
            num_parallel_calls=tf.data.AUTOTUNE
        ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return prepare_dataset(train_ds), prepare_dataset(val_ds), class_names