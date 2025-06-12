import tensorflow as tf
import os

# Parametry
image_size = (224, 224)
batch_size = 32
dataset_path = "C:/Users/david/Downloads/frames"  # např. složky cat/, dog/

# Načtení datasetu
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    image_size=image_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Třídy:", class_names)

# Předzpracování
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# Definice modelu
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, epochs=5)

# Ulož model (SavedModel formát)
model.save("saved_model")

# Ulož štítky
with open("labels.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")
