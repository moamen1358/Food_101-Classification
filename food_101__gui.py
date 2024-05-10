import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import os

# Global variables
selected_image = None
file_path = None
model = tf.keras.models.load_model("/home/moamen/Desktop/food_101_classification/covid_acc_98 (1).keras")




# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Function to predict class label
def predict_class(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class = food_classes = [
        "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", "beet_salad",
        "beignets", "bibimbap", "bread_pudding", "breakfast_burrito", "bruschetta", "caesar_salad",
        "cannoli", "caprese_salad", "carrot_cake", "ceviche", "cheesecake", "cheese_plate",
        "chicken_curry", "chicken_quesadilla", "chicken_wings", "chocolate_cake", "chocolate_mousse",
        "churros", "clam_chowder", "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame",
        "cup_cakes", "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots",
        "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries", "french_onion_soup",
        "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi",
        "greek_salad", "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gyoza", "hamburger",
        "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
        "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
        "mussels", "nachos", "omelette", "onion_rings", "oysters", "pad_thai", "paella", "pancakes",
        "panna_cotta", "peking_duck", "pho", "pizza", "pork_chop", "poutine", "prime_rib",
        "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi",
        "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara",
        "spring_rolls", "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu",
        "tuna_tartare", "waffles"
    ][predicted_class_index]
    return predicted_class

# Function to load and display the image
def load_image():
    global selected_image, file_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        selected_image = Image.open(file_path)
        selected_image.thumbnail((300, 300))  # Resize image to fit in GUI
        photo = ImageTk.PhotoImage(selected_image)
        label.config(image=photo)
        label.image = photo  # Keep a reference to the image to prevent garbage collection
        predict_button.config(state=tk.NORMAL)  # Enable predict button
        
        # Save the image in the same directory as the code
        code_directory = os.path.dirname(os.path.abspath(__file__))
        image_filename = os.path.basename(file_path)
        save_path = os.path.join(code_directory, image_filename)
        selected_image.save(save_path)
        print(f"Image saved at: {save_path}")

# Function to predict and display the result
def predict_image():
    global file_path
    if selected_image and file_path:
        class_label = predict_class(file_path)
        prediction_label.config(text=f"Prediction: {class_label}", font=("Helvetica", 14, "bold"), fg="#009933")

# Create tkinter window
root = tk.Tk()
root.title("Food101 Classifier")
root.configure(bg="#e6f5ff")  # Light blue background

# Set width and height of the window
window_width = 450
window_height = 500
root.geometry(f"{window_width}x{window_height}")

# Create a label to display the image
label = tk.Label(root, text="Upload an Food Image", font=("Helvetica", 16, "bold"), bg="#156511")  # Light blue background
label.pack(expand=True, fill='both', padx=10, pady=10)

# Create a button to select an image file
select_button = tk.Button(root, text="Select Image", font=("Helvetica", 12), command=load_image, bg="#66ccff", fg="white")
select_button.pack(pady=10)

# Create a button to predict
predict_button = tk.Button(root, text="Predict", font=("Helvetica", 12), command=predict_image, state=tk.DISABLED, bg="#009933", fg="white")
predict_button.pack(pady=10)

# Create a label to display prediction result
prediction_label = tk.Label(root, text="", font=("Helvetica", 14), bg="#e6f5ff")  # Light blue background
prediction_label.pack(pady=10)

# Run the tkinter event loop
root.mainloop()
