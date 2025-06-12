import tensorflow as tf
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

# --- Pomocné funkce ---

def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    scale = max_dim / max(shape)
    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    # VRACÍME tvar (height, width, 3) BEZ batch dimenze
    return img

def deprocess(img):
    # Normalizujeme do rozsahu 0-255 a převedeme na uint8
    img = 255 * (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
    return tf.cast(img, tf.uint8)

def show(img, title=None):
    if len(img.shape) == 4:
        img = tf.squeeze(img, axis=0)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# --- DeepDream část ---

# Načteme model InceptionV3 bez klasifikační hlavičky
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

# Vybereme vrstvy, jejichž aktivace chceme maximalizovat
names = ['mixed3', 'mixed5']
layers = [base_model.get_layer(name).output for name in names]

# Vytvoříme model, který na vstupu má obrázek a na výstupu aktivace těchto vrstev
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

def calc_loss(img, model):
    # img má tvar (height, width, 3), model chce (batch, h, w, 3)
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if not isinstance(layer_activations, list):
        layer_activations = [layer_activations]
    losses = []
    for act in layer_activations:
        losses.append(tf.reduce_mean(act))
    return tf.reduce_sum(losses)

@tf.function
def deepdream_step(img, model, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = calc_loss(img, model)
    gradients = tape.gradient(loss, img)
    gradients /= tf.math.reduce_std(gradients) + 1e-8
    img = img + gradients * learning_rate
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img

def run_deep_dream(img, model, steps=100, learning_rate=0.01):
    img = tf.convert_to_tensor(img)
    for step in range(steps):
        img = deepdream_step(img, model, learning_rate)
        if step % 10 == 0:
            print(f'Step {step}, loss computed')
    return img

# --- Hlavní program ---

input_path = "C:/Users/david/Downloads/frames/2/3.png"  # Cesta k obrázku
original_img = load_img(input_path)

dream_img = run_deep_dream(original_img, dream_model, steps=100, learning_rate=0.01)

show(deprocess(dream_img), 'DeepDream result')
