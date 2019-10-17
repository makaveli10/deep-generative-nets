import tensorflow as tf


def downsample(filters, size, apply_batchnorm=True):
    init = tf.random_normal_initializer(0., 0.02)
    down_result = tf.keras.Sequential()
    down_result.add(tf.keras.layers.Conv2D(filters,
                                           size,
                                           padding='same',
                                           strides=2,
                                           kernel_initializer=init,
                                           use_bias=False))
    if apply_batchnorm:
        down_result.add(tf.keras.layers.BatchNormalization())
    
    down_result.add(tf.keras.layers.LeakyReLU())
    return down_result


def upsample(filters, size, apply_dropout=True):
    init = tf.random_normal_initializer(0., 0.02)

    up_result = tf.keras.Sequential()

    up_result.add(tf.keras.layers.Conv2DTranspose(filters,
                                                  size,
                                                  strides=2,
                                                  padding='same',
                                                  kernel_initializer=init,
                                                  use_bias=False))
    
    up_result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        up_result.add(tf.keras.layers.Dropout(0.5))

    up_result.add(tf.keras.layers.LeakyReLU())
    return up_result


def Generator():
    down_stack = [
        downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
        downsample(128, 4), # (bs, 64, 64, 128)
        downsample(256, 4), # (bs, 32, 32, 256)
        downsample(512, 4), # (bs, 16, 16, 512)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4), # (bs, 1, 1, 512)
    ]

    up_stack = [
        # (1024 because of skip connection concatenation)
        upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024) 
        upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4), # (bs, 16, 16, 1024)
        upsample(256, 4), # (bs, 32, 32, 512)
        upsample(128, 4), # (bs, 64, 64, 256)
        upsample(64, 4), # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh') # (bs, 256, 256, 3)

    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape=[None,None,3])
    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
      x = down(x)
      skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
      x = up(x)
      x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    input_= tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
    target_ = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')

    x = tf.keras.layers.concatenate([input_, target_])  # (bs, 256, 256, channels*2)

    down_1 = downsample(64, 4, False)(x)    # (bs, 128, 128, 64)
    down_2 = downsample(128, 4)(down_1)     # (bs, 64, 64, 128)
    down_3 = downsample(256, 4)(down_2)     # (bs, 32, 32, 256)

    zero_pad_1 = tf.keras.layers.ZeroPadding2D()(down_3)      # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 
                                  4,
                                  strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad_1)       # (bs, 31, 31, 512)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad_2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)      # (bs, 33, 33, 512)
    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad_2)       #(bs, 30, 30, 1)

    return tf.keras.Model(inputs=[input_, target_], outputs=[last])

