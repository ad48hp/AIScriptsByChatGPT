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
LAMBDA_GP = 10.0  # váha gradient penalty

# --- Cesta k datasetu ---
DATA_DIR = "C:/Users/david/Downloads/frames/2"

# --- Augmentace ---
def augment(images):
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)
    images = tf.image.random_brightness(images, max_delta=0.1)
    images = tf.image.random_contrast(images, lower=0.9, upper=1.1)
    images = tf.image.random_saturation(images, lower=0.9, upper=1.1)
    return images

# --- Načtení a předzpracování datasetu ---
dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    label_mode=None,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

def normalize_and_augment(images):
    images = tf.cast(images, tf.float32)
    images = augment(images)
    images = (images / 127.5) - 1.0  # normalizace do [-1,1]
    return images

train_dataset = dataset.map(normalize_and_augment).prefetch(tf.data.AUTOTUNE)

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

        layers.Conv2D(128, (5,5), strides=(2,2), padding='same'),
        layers.LeakyReLU(0.2),

        layers.Conv2D(256, (5,5), strides=(2,2), padding='same'),
        layers.LeakyReLU(0.2),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

# --- Loss funkce ---
def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

def critic_loss(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

# --- Gradient penalty ---
def gradient_penalty(critic, real_images, fake_images):
    batch_size = real_images.shape[0]
    epsilon = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated = epsilon * real_images + (1 - epsilon) * fake_images

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = critic(interpolated, training=True)

    grads = gp_tape.gradient(pred, [interpolated])[0]
    grads = tf.reshape(grads, [batch_size, -1])
    grads_norm = tf.norm(grads, axis=1)
    gp = tf.reduce_mean((grads_norm - 1.0) ** 2)
    return gp

# --- Inicializace ---
generator = make_generator_model()
critic = make_critic_model()

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.9)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.9)

# --- Tréninkový krok ---
@tf.function
def train_step(real_images):
    batch_size = tf.shape(real_images)[0]  # dynamická velikost batch
    noise = tf.random.normal([batch_size, NOISE_DIM])

    for _ in range(5):
        with tf.GradientTape() as crit_tape:
            generated_images = generator(noise, training=True)
            real_output = critic(real_images, training=True)
            fake_output = critic(generated_images, training=True)

            gp = gradient_penalty(critic, real_images, generated_images)
            crit_loss = critic_loss(real_output, fake_output) + LAMBDA_GP * gp

        gradients_of_critic = crit_tape.gradient(crit_loss, critic.trainable_variables)
        critic_optimizer.apply_gradients(zip(gradients_of_critic, critic.trainable_variables))

    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = critic(generated_images, training=True)
        gen_loss = generator_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return crit_loss, gen_loss
    

# --- Ukládání generovaných obrázků ---
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    predictions = (predictions + 1.0) / 2.0  # [0,1]

    fig = plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4,4,i+1)
        plt.imshow(predictions[i])
        plt.axis('off')

    os.makedirs('generated_images_augmentedwasserstein', exist_ok=True)
    plt.savefig(f"generated_images_augmentedwasserstein/image_at_epoch_{epoch:04d}.png")
    plt.close()

# --- Trénink ---
def train(dataset, epochs):
    seed = tf.random.normal([16, NOISE_DIM])
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            crit_loss, gen_loss = train_step(image_batch)

        # if (epoch + 1) % 10 == 0:
        generate_and_save_images(generator, epoch + 1, seed)

        print(f"Epoch {epoch+1}, Critic loss: {crit_loss:.4f}, Generator loss: {gen_loss:.4f}, Time: {time.time()-start:.2f}s")

# --- Spuštění tréninku ---
train(train_dataset, EPOCHS)
