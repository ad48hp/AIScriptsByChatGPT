import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import time
from matplotlib import pyplot as plt

# --- Hyperparametry ---
IMG_SIZE = 128
CHANNELS = 3
NOISE_DIM = 100
BATCH_SIZE = 16
EPOCHS = 100
CLIP_VALUE = 0.01
CRITIC_ITERATIONS = 5

# --- Cesta k datasetu ---
DATA_DIR = "C:/Users/david/Downloads/frames/2"

# --- Načtení datasetu ---
dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    label_mode=None,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Normalizace obrázků na [-1, 1]
def normalize(images):
    images = tf.cast(images, tf.float32)
    images = (images / 127.5) - 1.0
    return images

train_dataset = dataset.map(normalize).prefetch(buffer_size=tf.data.AUTOTUNE)

# --- Generator ---
def make_generator_model():
    model = tf.keras.Sequential([
        layers.Dense(8*8*256, use_bias=False, input_shape=(NOISE_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((8, 8, 256)),

        layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(32, (5,5), strides=(2,2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(CHANNELS, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

# --- Critic ---
def make_critic_model():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5,5), strides=(2,2), padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)  # bez sigmoid aktivace!
    ])
    return model

# --- Loss funkce ---
def critic_loss(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

# --- Inicializace ---
generator = make_generator_model()
critic = make_critic_model()

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.9)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.9)

# --- Tréninkový krok ---
@tf.function
def train_step(real_images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    # Trénink kritika několikrát
    for _ in range(CRITIC_ITERATIONS):
        with tf.GradientTape() as crit_tape:
            generated_images = generator(noise, training=True)
            real_output = critic(real_images, training=True)
            fake_output = critic(generated_images, training=True)
            crit_loss = critic_loss(real_output, fake_output)

        gradients_of_critic = crit_tape.gradient(crit_loss, critic.trainable_variables)
        critic_optimizer.apply_gradients(zip(gradients_of_critic, critic.trainable_variables))

        # Weight clipping
        for var in critic.trainable_variables:
            var.assign(tf.clip_by_value(var, -CLIP_VALUE, CLIP_VALUE))

    # Trénink generátoru
    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = critic(generated_images, training=True)
        gen_loss = generator_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return crit_loss, gen_loss

# --- Pomocná funkce pro generování a ukládání obrázků ---
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    predictions = (predictions + 1.0) / 2.0  # z [-1,1] na [0,1]

    fig = plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4,4,i+1)
        plt.imshow(predictions[i])
        plt.axis('off')

    os.makedirs('generated_images_wasserstein', exist_ok=True)
    plt.savefig(f"generated_images_wasserstein/image_at_epoch_{epoch:04d}.png")
    plt.close()

# --- Tréninková smyčka ---
def train(dataset, epochs):
    seed = tf.random.normal([16, NOISE_DIM])

    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            crit_loss, gen_loss = train_step(image_batch)

        # Uložit obrázky každých 10 epoch
        # if (epoch + 1) % 10 == 0:
        generate_and_save_images(generator, epoch + 1, seed)

        print(f"Epoch {epoch+1}, Critic loss: {crit_loss:.4f}, Generator loss: {gen_loss:.4f}, Time: {time.time()-start:.2f}s")

# --- Spustit trénink ---
train(train_dataset, EPOCHS)
