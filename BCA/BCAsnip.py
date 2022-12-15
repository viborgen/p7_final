def spatial_block_1(x):
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def color_block_1(x):
    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def bca_block1(x_spatial, x_color):
    x = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, padding="same")(x_color)
    x = tf.keras.layers.Activation('sigmoid')(x)
    x_spatial_new = tf.math.multiply(x_spatial, x)
    x = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, padding="same")(x_spatial)
    x = tf.keras.layers.Activation('sigmoid')(x)
    x_color_new = tf.math.multiply(x_color, x)
    return x_spatial_new, x_color_new

def spatial_block_2(x):
    x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def color_block_2(x):
    x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def bca_block2(x_spatial, x_color):
    x = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")(x_color)
    x = tf.keras.layers.Activation('sigmoid')(x)
    x_spatial_new = tf.math.multiply(x_spatial, x)
    x = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")(x_spatial)
    x = tf.keras.layers.Activation('sigmoid')(x)
    x_color_new = tf.math.multiply(x_color, x)
    return x_spatial_new, x_color_new

def concatenate_block(x_spatial, x_color):
    x = tf.keras.layers.Concatenate(axis=-1)([x_spatial, x_color])
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x


# BCA IMPLEMENTATION
def BCAImpl(shapeSpatial, shapeColor, classes):
    # Step 1 (Setup Input Layer)
    x_color = tf.keras.layers.Input(shape = shapeColor, name = "x_color_input")
    x_spatial = tf.keras.layers.Input(shape = shapeSpatial, name = "x_spatial_input")

    x_spatial_input = x_spatial
    x_color_input = x_color

    x_spatial = spatial_block_1(x_spatial_input)
    x_color = color_block_1(x_color_input)
    x_spatial, x_color = bca_block1(x_spatial, x_color)
    x_spatial = spatial_block_2(x_spatial)
    x_color = color_block_2(x_color)
    x_spatial, x_color = bca_block2(x_spatial, x_color)
    output = concatenate_block(x_spatial, x_color)

    #Residual Network
    x = tf.keras.layers.ZeroPadding2D((3, 3))(output)
    #x = data_augmentation(x)
    #x = tf.keras.layers.GaussianNoise(0.1)(x)
    # Step 2 (Initial Conv layer along with maxPool)
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    
    #Block1
    #Todo: s=1
    x = convolution_block_res101(x,f=3,filters=[64,64,128],s=1)
    for i in range(1):
        x = identity_block_res101(x,3,[64,64,128])

    #Block2
    x = convolution_block_res101(x,f=3,filters=[128,128,512],s=2)
    for i in range(2):
         x = identity_block_res101(x,3,[128,128,512])

    #Block3
    x = convolution_block_res101(x,f=3,filters=[256,256,1024],s=2)
    for i in range(21):
         x = identity_block_res101(x,3,[256,256,1024])

    #Block4
    x = convolution_block_res101(x,f=3,filters =[512,512,2048],s=2)
    for i in range(1):
        x = identity_block_res101(x,3,[512,512,2048])

    # Step 4 End Dense Network
    x = tf.keras.layers.AveragePooling2D((2,2), padding = 'same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation = 'relu')(x)
    x = tf.keras.layers.Dense(classes, activation = 'softmax')(x)
    model = tf.keras.models.Model(inputs = [x_spatial_input, x_color_input], outputs = x, name = "ResNet101")
    return model