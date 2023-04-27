import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import MaxPooling2D, UpSampling2D
from keras.models import Model
from numpy import reshape
import matplotlib.pyplot as plt

xtrain = tf.keras.utils.image_dataset_from_directory(
    directory = "C://Users//jeanc//Cours CS//ST2 Observer la Terre//EI Images//image_train",
    labels=None,
    label_mode=None,
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(32, 32),
    shuffle=False,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)


auto_train = xtrain.map(lambda x: (x, x)).reshape(32,32,3)

input_img = Input(shape=(32, 32, 3))

enc_conv1 = Conv2D(12, (3, 3), activation='relu', padding='same')(input_img)
enc_pool1 = MaxPooling2D((2, 2), padding='same')(enc_conv1)
enc_conv2 = Conv2D(10, (4, 4), activation='relu', padding='same')(enc_pool1)
enc_pool2 = MaxPooling2D((4, 4), padding='same')(enc_conv2)
enc_conv3 = Conv2D(8, (4, 4), activation='relu', padding='same')(enc_pool2)
enc_ouput = MaxPooling2D((4, 4), padding='same')(enc_conv2)

dec_conv2 = Conv2D(8, (4, 4), activation='relu', padding='same')(enc_ouput)
dec_upsample2 = UpSampling2D((4, 4))(dec_conv2)
dec_conv3 = Conv2D(12, (3, 3), activation='relu', padding='same')(dec_upsample2)
dec_upsample3 = UpSampling2D((2, 2))(dec_conv3)
dec_conv4 = Conv2D(10, (3, 3), activation='relu', padding='same')(dec_upsample3)
dec_upsample4 = UpSampling2D((2, 2))(dec_conv3)
dec_output = Conv2D(3, (3, 3), activation='relu', padding='same')(dec_upsample4)

autoencoder = Model(input_img, dec_output)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.summary()
 
autoencoder.fit(auto_train, epochs=1, batch_size=128, shuffle=True)

decoded_imgs = autoencoder.predict(auto_train)
print(decoded_imgs.shape)
n = 10
batch = next(iter(xtrain))
fig, axes = plt.subplots(2, n, figsize=(20, 4))
for i in range(n):
    #affiche les images décodées par le réseau CAE
    axes[1, i].imshow(decoded_imgs[i].astype("uint8"))
    batch_img = batch[i].numpy().astype("uint8")
    axes[0, i].imshow(batch_img)
plt.show()

autoencoder.save("C://Users//jeanc//Cours CS//ST2 Observer la Terre//EI//Model")

