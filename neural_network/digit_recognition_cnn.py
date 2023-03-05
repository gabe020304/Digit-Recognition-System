import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.utils import load_img, img_to_array
from keras.models import load_model
from utils import check_model_exists, plot_history


# Load the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape the data to fit the model
X_train = np.reshape(X_train, (-1, 28, 28, 1))
X_test = np.reshape(X_test, (-1, 28, 28, 1))

# Convert the labels to categorical variables
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Create the model
model = Sequential()

# Convolutional layer 1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Pooling layer 1
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional layer 2
model.add(Conv2D(64, (3, 3), activation='relu'))

# Pooling layer 2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten layer
model.add(Flatten())

# Fully connected layer 1
model.add(Dense(128, activation='relu'))

# Dropout layer
model.add(Dropout(0.5))

# Output layer
model.add(Dense(10, activation='softmax'))

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Provide training data
history = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=0.2)

print(os.getcwd())  # Check current working directory
model = load_model('digit_recognition_cnn.h5')

model = check_model_exists(model, history)

# Load and preprocess the test image
img = load_img('test_image.png', color_mode='grayscale', target_size=(28, 28))
img_arr = img_to_array(img)
img_arr = img_arr / 255.0
img_arr = img_arr.reshape(1, 28, 28, 1)

# Use the model to predict the digit
prediction = model.predict(img_arr)
digit = np.argmax(prediction)

print('Predicted digit:', digit)

# Call the plot_history function after training is done
plot_history(history)
