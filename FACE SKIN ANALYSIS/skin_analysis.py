import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('models/skin_model.h5')
labels = ['Acne', 'Eye Bags', 'Hyperpigmentation', 'Wrinkles', 'Clear']

def analyze_skin(face_image):
    resized = cv2.resize(face_image, (128, 128))  # Depends on model input
    normalized = resized / 255.0
    input_data = np.expand_dims(normalized, axis=0)

    predictions = model.predict(input_data)
    predicted_index = np.argmax(predictions)
    return labels[predicted_index]
