def identity_block_res101(x,f,filters):
    x_skip = x

    F1, F2, F3 = filters

    #Layer 1
    x = tf.keras.layers.Conv2D(F1,kernel_size = (1,1), padding = 'valid')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    #Layer 2
    x = tf.keras.layers.Conv2D(F2,kernel_size = (f,f), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    #Layer3
    x = tf.keras.layers.Conv2D(F3, kernel_size = (1,1), padding= 'valid')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)

    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    return x

def convolution_block_res101(x,f,filters,s=2):
    x_skip = x

    F1, F2, F3 = filters
    #Layer1
    x = tf.keras.layers.Conv2D(filters = F1, kernel_size = (1,1), strides = (s,s),padding = 'valid')(x)
    x = tf.keras.layers.BatchNormalization(axis = 3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    #Layer2
    x = tf.keras.layers.Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1),padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis = 3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    #Layer3
    x = tf.keras.layers.Conv2D(filters = F3, kernel_size = (1,1), strides = (s,s),padding = 'valid')(x)
    x = tf.keras.layers.BatchNormalization(axis = 3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    #Shortcut projection
    x_skip = tf.keras.layers.Conv2D(filters = F3, kernel_size = (1,1), strides = (s,s),padding = 'valid')(x_skip)

    x_skip = tf.keras.layers.BatchNormalization(axis = 3)(x)

    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    return x


def ResNet101(shape,classes):
    # Step 1 (Setup Input Layer)
    x_input = tf.keras.layers.Input(shape)
    x = tf.keras.layers.ZeroPadding2D((3, 3))(x_input)
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    
    #Sub-Block1
    x = convolution_block_res101(x,f=3,filters=[64,64,128],s=1)
    for i in range(1):
        x = identity_block_res101(x,3,[64,64,128])

    #Sub-Block2
    x = convolution_block_res101(x,f=3,filters=[128,128,512],s=2)
    for i in range(2):
         x = identity_block_res101(x,3,[128,128,512])

    #Sub-Block3
    x = convolution_block_res101(x,f=3,filters=[256,256,1024],s=2)
    for i in range(21):
         x = identity_block_res101(x,3,[256,256,1024])

    #Sub-Block4
    x = convolution_block_res101(x,f=3,filters =[512,512,2048],s=2)
    for i in range(1):
        x = identity_block_res101(x,3,[512,512,2048])

    # Step 4 End Dense Network
    x = tf.keras.layers.AveragePooling2D((2,2), padding = 'same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation = 'relu')(x)
    x = tf.keras.layers.Dense(classes, activation = 'softmax')(x)
    model = tf.keras.models.Model(inputs = x_input, outputs = x, name = "ResNet101")
    return model