import tensorflow as tf
import numpy as np
from PIL import Image

# Cesty k modelu a souborům
MODEL_PATH = "saved_model"
LABELS_PATH = "labels.txt"
IMAGE_PATH = "C:/Users/david/Downloads/frames/2/3.png"
IMAGE_SIZE = (224, 224)

# Načtení modelu a labelů
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f]

# Načti a předzpracuj obrázek
def load_and_preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0  # Normalizace
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimenze
    return img_array

image = load_and_preprocess_image(IMAGE_PATH)

# Předpověď
predictions = model.predict(image)[0]
predicted_index = np.argmax(predictions)
predicted_label = class_names[predicted_index]
confidence = predictions[predicted_index]

# Výstup
print(f"Predikce: {predicted_label} ({confidence*100:.2f}%)")
