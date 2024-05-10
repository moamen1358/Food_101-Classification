from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import cv2
import numpy as np
import shutil
import os
import tensorflow as tf

app = FastAPI()

# Load the trained model
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
    predicted_class = [
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

# Function to save and process uploaded image
def save_and_process_image(file):
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    processed_image_path = f"processed_{file.filename}"
    image = Image.open(file.filename)
    image.thumbnail((300, 300))
    image.save(processed_image_path)
    return processed_image_path

# Prediction endpoint
@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        processed_image_path = save_and_process_image(image)
        class_label = predict_class(processed_image_path)
        os.remove(processed_image_path)
        return {"prediction": class_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    