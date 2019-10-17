from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import os
import time
import matplotlib.pyplot as plt

from utils import *
from model import Generator, Discriminator


BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


def load_img_train(image_file):
    inp, re = load_image(image_file)
    inp, re = random_jitter(inp, re)
    inp, re = normalize(inp, re)
    return inp, re


def load_img_test(image_file):
    inp, re = load_image(image_file)
    inp, re = resize_image(inp, re, IMG_HEIGHT, IMG_WIDTH)
    inp, re = normalize(inp, re)
    return inp, re


def generate_images(model, inp, tar):
    prediction = model(inp, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [inp[0], prediction[0], tar[0]]
    labels = ['input', 'predicted_image', 'ground_truth']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(labels[i])

        # get the pixel values between [0, 1] to plot them
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'

path_to_zip = tf.keras.utils.get_file('facades.tar.gz',
                                      origin=_URL,
                                      extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')
# INPUT PIPELINE
# train dataset
train_dataset = tf.data.Dataset.list_files(PATH+'train/*.jpg')
train_dataset = train_dataset.map(load_img_train, 
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(1)
# test dataset
test_dataset = tf.data.Dataset.list_files(PATH+'test/*.jpg')
test_dataset = test_dataset.map(load_img_test)
test_dataset = test_dataset.batch(1)
generator = Generator()
discriminator = Discriminator()
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(disc_real_output, disc_gen_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    gen_loss = loss_object(tf.zeros_like(disc_gen_output), disc_gen_output)
    total_loss = real_loss + gen_loss
    return total_loss
def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    # MAE
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    loss = gan_loss + (LAMBDA * l1_loss)
    return loss
gen_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
dis_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
checkpoint_dir = './pix2pix_training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(gen_optimizer=gen_optimizer,
                                     dis_optimizer=dis_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
                                     
@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)
    
    gen_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    dis_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))


def train():
    for epoch in range(epochs):
        start = time.time()

        # Train
        for input_image, target in train_ds:
            train_step(input_image, target)

        # Test on the same image so that the progress of the model can be 
        # easily seen.
        for example_input, example_target in test_ds.take(1):
            generate_images(generator, example_input, example_target)

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
          checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                          time.time()-start))   


if __name__=='__main__':
    train()