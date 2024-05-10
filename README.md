# Food 101 Classification Model

## Overview

This repository contains an AI model trained to classify food images into 101 different classes. The model achieves an accuracy of 80% on the test dataset.

## Model Architecture

The model is built using transfer learning, leveraging a pre-trained convolutional neural network (CNN) architecture. Transfer learning enables us to utilize knowledge gained from training on a large dataset (typically ImageNet) and apply it to our specific classification task. We fine-tune the pre-trained model on our dataset containing food images.

## Dataset

The dataset used for training and evaluation consists of food images collected from the Food-101 dataset. It contains images categorized into 101 food classes. The dataset is split into training, validation, and test sets to train and evaluate the model's performance.

## Training

During training, we employ transfer learning by initializing the model with pre-trained weights. We fine-tune the model on our dataset using techniques such as data augmentation and adjusting learning rates to enhance generalization and achieve a satisfactory level of accuracy.

## Evaluation

The model's performance is evaluated on a separate test dataset that was not seen during training or validation. We compute metrics such as accuracy, precision, recall, and F1-score to assess the model's ability to correctly classify food images into their respective classes.

## Results

The model achieves an accuracy of 80% on the test dataset. Here are the detailed evaluation metrics:

Accuracy: 80%

# installation guide

1. create new conda environment

```bash
conda create -n food101 python=3.12
```

2. activate the enviroment

```bash
conda activate covid
```

3. install the packages needed for the project

```bash
pip install -r requirements.txt
```

4. Start FastAPI Application

```bash
uvicorn main:app
```

## Usage

### Using Postman to Upload an Image

1. Open Postman.
2. Create a new request.
3. Set the request method to POST.
4. Enter the URL of your FastAPI application along with the specific endpoint that expects the image.
   - Example URL: `http://localhost:8000/predict`
5. Click on the "Body" tab.
6. Choose "form-data" as the body type.
7. Add a key-value pair where the key corresponds to the name of the parameter expected by your API for the image, and the value is the image file you want to upload.
   - Key: `image`
   - Value: [Choose Files] button to select the image file from your local system.
8. Click the "Send" button to send the request to your FastAPI application.
