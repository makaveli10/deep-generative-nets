import tensorflow as tf


# load image
def load_image(image_file):
    # read image_file
    image = tf.io.read_file(image_file)
    
    # decode the read image file
    image = tf.io.decode_jpeg(image)

    #get width of image
    w = tf.shape(image)[1]

    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


def resize_image(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width], 
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width], 
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_images = tf.stack([input_image, real_image], axis=0)
    # Randomly cropping a tensor to a given size.
    cropped_images = tf.image.random_crop(stacked_images, 
                                          size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_images[0], cropped_images[1]


# normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image


def random_jitter(input_image, real_image):
    # resizing to 286 x 286 x 3
    input_image, real_image = resize_image(input_image, real_image, 286, 286)

    # randomly crop to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)

    # random mirroring
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
    
    return input_image, real_image