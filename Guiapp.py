import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
file_path = ""
class_labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]
#load model
loaded_model_RF = pickle.load(open('RF_Model_MNIST.sav', 'rb'))


#make the predict function
def predict():
    img= cv2.imread(file_path,0)
    img = cv2.resize(img, (28,28), interpolation = cv2.INTER_AREA)
    img = cv2.bitwise_not(img)
    plt.imshow(img)
    plt.show()
    img = np.reshape(list(img),(1,-1))
    global label
    s = class_labels[loaded_model_RF.predict(img)[0]]
    if hasattr(predict, "label"):
        predict.label.config(text=s,font=("Helvetica", 32))
    else:
        predict.label = tk.Label(predict_frame, text="Label in Predict")
        predict.label.pack() 

def insert_image():
    global file_path
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image = image.resize((500, 500), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo,)
        label.image = photo

# Create the main window
root = tk.Tk()
root.title("Image Viewer")
root.geometry("1000x500")

# Create a label to display the image at the top
label = tk.Label(root)
label.pack(side="top", fill="both", expand="true")

# Create a frame to hold the "Insert Image" button and "Predict" label
predict_frame = tk.Frame(root)
predict_frame.pack(side="top", pady=10)

# Create a "Predict" label (initially empty)
predict.label = tk.Label(predict_frame, text="")
predict.label.pack()

# Create an "Insert Image" button with a larger size
insert_button = tk.Button(predict_frame, text="Insert Image", command=insert_image, width=20, height=2)
insert_button.pack()

# Create a "Predict" button with a larger size
predict_button = tk.Button(root, text="Predict", command=predict, width=20, height=2)
predict_button.pack(side="top", pady=10)

# Start the main loop
root.mainloop()