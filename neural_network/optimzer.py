from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from digit_recognition_cnn import model
from usps_dataset import usps

# compile the model
model.compile(loss=SparseCategoricalCrossentropy(), optimizer=Adam(), metrics=[SparseCategoricalAccuracy()])

# train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, y_val))
