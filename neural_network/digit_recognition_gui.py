import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk, Image
import requests
from io import BytesIO
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('neural_network/digit_recognition_cnn.h5')


class DigitRecognitionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Digit Recognition AI")

        # Create the input label and text entry box
        input_label = tk.Label(self.root, text="Enter image URL:")
        input_label.pack()
        self.input_entry = tk.Entry(self.root, width=50)
        self.input_entry.pack()

        # Create the predict button
        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict)
        self.predict_button.pack()

        # Create the output label and image display area
        self.output_label = tk.Label(self.root, text="")
        self.output_label.pack()
        self.image_canvas = tk.Canvas(self.root, width=200, height=200)
        self.image_canvas.pack()

    def predict(self):
        # Get the image URL from the text entry box
        url = self.input_entry.get()

        try:
            # Load the image from the URL and convert it to a NumPy array
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert('L')
            img = img.resize((28, 28))
            img_arr = np.array(img)

            # Preprocess the image
            img_arr = img_arr / 255.0
            img_arr = img_arr.reshape(1, 28, 28, 1)

            # Use the model to predict the digit
            prediction = model.predict(img_arr)
            digit = np.argmax(prediction)

            # Display the predicted digit and image
            self.output_label.config(text="Predicted digit: {}".format(digit))
            img_tk = ImageTk.PhotoImage(img)
            self.image_canvas.create_image(100, 100, image=img_tk)
            self.image_canvas.image = img_tk
        except Exception as e:
            # Display an error message if there was a problem with the URL or image
            messagebox.showerror("Error", "Error loading image from URL: {}".format(e))

    def run(self):
        self.root.mainloop()

# Create an instance of the GUI application and run it


app = DigitRecognitionGUI()
app.run()
