from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from usps_dataset import X_train, y_train, X_val, y_val


def get_data_augmentation():
    # Define data augmentation function
    data_augmentation = ImageDataGenerator(rotation_range=20,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)
    return data_augmentation


class DigitRecognitionCNN:
    sequential_model = Sequential()

    def __init__(self, input_shape, num_classes):
        print("Initializing DigitRecognitionCNN optimizer")

        # Define the model architecture
        self.sequential_model = Sequential()
        self.sequential_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        self.sequential_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.sequential_model.add(Flatten())
        self.sequential_model.add(Dense(num_classes, activation='softmax'))

    def train_model(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, data_augmentation=None):
        # Compile the model
        self.sequential_model.compile(loss=SparseCategoricalCrossentropy(),
                                      optimizer=Adam(),
                                      metrics=[SparseCategoricalAccuracy()])

        # Create data generator for training set with optional data augmentation
        if data_augmentation is not None and isinstance(data_augmentation, ImageDataGenerator):
            train_datagen = ImageDataGenerator(preprocessing_function=data_augmentation)
        else:
            train_datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1,
                                               height_shift_range=0.1, shear_range=0.1, fill_mode='nearest')

        train_datagen.fit(X_train)

        # Create data generator for validation set
        if data_augmentation is not None and isinstance(data_augmentation, ImageDataGenerator):
            val_datagen = ImageDataGenerator(preprocessing_function=data_augmentation)
        else:
            val_datagen = ImageDataGenerator()

        val_datagen.fit(X_val)

        # Train the model
        history = self.sequential_model.fit(train_datagen.flow(X_train, y_train, batch_size=batch_size),
                                            steps_per_epoch=len(X_train) // batch_size,
                                            epochs=epochs,
                                            validation_data=val_datagen.flow(X_val, y_val, batch_size=batch_size),
                                            validation_steps=len(X_val) // batch_size)

        return self.sequential_model, history


# Create an instance of the DigitRecognitionCNN class
model = DigitRecognitionCNN(input_shape=(16, 16, 1), num_classes=11)

# Train the model with data augmentation
model.train_model(X_train, y_train, X_val, y_val, epochs=10, batch_size=32, data_augmentation=get_data_augmentation())
