import numpy as np
from PIL import Image
import requests
from io import BytesIO
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.datasets import mnist
from keras.utils import to_categorical, img_to_array
from keras.models import model_from_json
from utils import plot_history


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

# Save the model in JSON format
model_json = model.to_json()
with open("digit_recognition_cnn.json", "w") as json_file:
    json_file.write(model_json)

# Save the model weights
model.save_weights("digit_recognition_cnn_weights.h5")

# Load the model from JSON format
json_file = open('digit_recognition_cnn.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load the model weights
model.load_weights("digit_recognition_cnn_weights.h5")

# Load and preprocess the test image
url = 'https://imgs.search.brave.com/uZubc-xmdO97OSZEJ0KVsxTVIZtMB5s-6RJ0QdE8jeY/rs:fit:477:225:1/g:ce/aHR0cHM6Ly90c2Uy/Lm1tLmJpbmcubmV0/L3RoP2lkPU9JUC5q/NlRZZ1V5OFRvVktP/M0YycXVaQ2pnSGFI/WCZwaWQ9QXBp'
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img = img.convert('L')  # convert to grayscale
img = img.resize((28, 28))  # resize to target size
img_ready = np.expand_dims(img_to_array(img), axis=0) / 255.0

# Use the model to predict the digit
prediction = model.predict(img_ready)
digit = np.argmax(prediction)

print('Predicted digit:', digit)

# Call the plot_history function after training is done
plot_history(history)
