import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

IMG_SIZE = 128
CHANNELS = 3
NOISE_DIM = 100
BATCH_SIZE = 16
EPOCHS = 200

# --- Data augmentation pipeline ---
def augment(images):
    images = tf.image.resize(images, [128, 128])  # zajistí správný rozměr
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_brightness(images, max_delta=0.1)
    images = tf.image.random_contrast(images, lower=0.9, upper=1.1)
    images = tf.image.random_saturation(images, lower=0.9, upper=1.1)
    return images


def load_and_preprocess(directory):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        label_mode=None,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='rgb',
        shuffle=True
    )
    dataset = dataset.map(lambda x: (x / 127.5) - 1.0)  # normalize [-1,1]
    dataset = dataset.map(lambda x: augment(x))
    return dataset.prefetch(tf.data.AUTOTUNE)

train_dataset = load_and_preprocess("C:/Users/david/Downloads/frames/2")

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


# --- Discriminator ---
def make_discriminator_model():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5,5), strides=(2,2), padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

# --- Loss + optimizers ---
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# --- Training step ---
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        print(generated_images.shape)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

# --- Generate & save images ---
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    predictions = (predictions + 1.0) / 2.0  # scale back to [0,1]

    fig = plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4,4,i+1)
        plt.imshow(predictions[i].numpy())
        plt.axis('off')
    plt.savefig(f'generated_images/epoch_{epoch:04d}.png')
    plt.close()

# Vytvoř složku pro augmented obrázky
os.makedirs("augmented", exist_ok=True)


# --- Training loop ---
def train(dataset, epochs):
    os.makedirs('generated_images', exist_ok=True)
    seed = tf.random.normal([16, NOISE_DIM])

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for image_batch in dataset:
            # print(image_batch.shape)
            train_step(image_batch)

        generate_and_save_images(generator, epoch+1, seed)

train(train_dataset, EPOCHS)
        