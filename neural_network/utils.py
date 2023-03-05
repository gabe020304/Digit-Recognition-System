import os
import matplotlib.pyplot as plt


def check_model_exists(model, history):
    # Check if the model file exists
    if os.path.exists('digit_recognition_cnn.h5'):
        # Delete the model file
        os.remove('digit_recognition_cnn.h5')

    # Save the model if accuracy is above 90%
    if history.history['val_accuracy'][-1] > 0.9:
        model.save('digit_recognition_cnn.h5')
        return model


def plot_history(history):
    # Plot the training accuracy and validation accuracy
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
