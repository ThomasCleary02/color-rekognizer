import tensorflow as tf
from data_preprocessing import prepare_datasets
from model import create_model, train_model
from prediction import predict_color

def main():
    """
    Main function to run the color classification model.
    """
    # Prepare datasets
    train_ds, val_ds, class_names = prepare_datasets()

    # Create and train the model
    model = create_model(len(class_names))
    trained_model = train_model(model, train_ds, val_ds)

    # Make a prediction
    predict_color(trained_model, "ColorClassification/testimg/17.jpg", class_names)

if __name__ == "__main__":
    main()