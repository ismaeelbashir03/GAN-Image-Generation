import tensorflow_datasets as tf_d
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.models import Model
import os
from keras_preprocessing.image import array_to_img
from keras.callbacks import Callback

# A gan is a generative adversarial network, this is a model where you use a generative network
# using cnn layers to create an image and compute its loss and train it with a discriminator
# network which will try to detect fake images, using this the generative model will learn
# to make more realistoc models as its loss depends on the discriminator.

# setting vram for the model
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

'''
# we load in our dataset but only the training images
# the dataset is an iterable pipeline of images, 
# (doesnt store all the memory at once)
dataset = tf_d.load('fashion_mnist',split = 'train')
#print(dataset.as_numpy_iterator().next().keys())

# getting the data as a iterable
data_iter = dataset.as_numpy_iterator()


# showing our images to see if they have loaded in correctly
fig, ax = plt.subplots(ncols=4)
# loop through each index of the subplot
for idx in range(4):
    # getting the next sample image
    sample = data_iter.next()
    ax[idx].imshow(np.squeeze(sample['image']))
    ax[idx].set_title(sample['label'])

plt.show()
'''

# scaling down our pixel values to be in between 0 and 1 to be fed into a nueral network
def scale_images(data):
    img = data['image']

    # dividing by 255 as thta is the max value of the pixels
    img = img / 255

    return img

# we load in our dataset but only the training images
# the dataset is an iterable pipeline of images, 
# (doesnt store all the memory at once)
ds = tf_d.load('fashion_mnist', split='train')

# we use a mapping function to run our scale images function on all the images
ds = ds.map(scale_images)

# we then cache our data
ds = ds.cache()

# we then have to shuffle our data, 60000 is our shuffle buffer
ds = ds.shuffle(60000)

# setting the batch to 128 images
ds = ds.batch(128)

# setting how much we prefetch by, reduces the chance of bottlenecking
ds = ds.prefetch(64)

'''
# checking to see if our batching has worked
print(ds.as_numpy_iterator().next().shape)
'''

# we can now build our generator
def build_generator():
    # initialising our model
    model = Sequential()

    # adding our first input layer, which expands our input to be bigger

    # we start with a Dense input layer, our output is multiplied by 7 twice as
    # we want to expand the input to be bigger so we can reshape into an image later
    model.add(Dense(units = 7*7*128, input_dim = 128))
    # we use a LeakyReLU as our activation function for this model as we want some negative
    # values, a leaky relu does this by slanting down the line across the x axis by the given
    # paramemter (_/) 
    model.add(LeakyReLU(0.2))
    # we can now reshape our output to start making it look like an image, we do this as
    # we continue down the layers, this output is a 7 by 7 image with a depth of 128
    model.add(Reshape((7,7,128)))

    # now we add our second layer to exapnd the output even more using upsampling
    
    # this layer is used to extend our dimensions of our input image by copying the 
    # current value across this results in dimension double of the previous
    model.add(UpSampling2D())
    # here we add our first convolutional layer which uses feature maps (matrices) to be 
    # dot producted with our image to identify features, we start lower level (edges),
    # here our feature maps are 5x5 and we output 128, we specify padding to be 
    # the same so the layer does not decrease the image dimensions
    model.add(Conv2D(128, 5, padding='same'))
    # here we add our activation function again
    model.add(LeakyReLU(0.2))

    # now we add our third layer to exapnd the output even more using upsampling
    
    # this layer is used to extend our dimensions of our input image by copying the 
    # current value across this results in dimension double of the previous
    model.add(UpSampling2D())
    # here we add our second convolutional layer which uses feature maps (matrices) to be 
    # dot producted with our image to identify features, we start lower level (edges),
    # here our feature maps are 5x5 and we output 128, we specify padding to be 
    # the same so the layer does not decrease the image dimensions
    model.add(Conv2D(128, 5, padding='same'))
    # here we add our activation function again
    model.add(LeakyReLU(0.2))

    # currently we could have the right dimensions to ouput our image if we copy the same layer again
    # as we would have an output of 28x28x1, but currently our model is not complex enough to create images that 
    # are like our training data, so lets add more layers

    # we can add our fourth layer to compute more features without upsampling

    # here we add our fourth convolutional layer which uses feature maps (matrices) to be 
    # dot producted with our image to identify features,
    # here our feature maps are 4x4 and we output 128, we specify padding to be 
    # the same so the layer does not decrease the image dimensions
    model.add(Conv2D(128, 4, padding='same'))
    # here we add our activation function again
    model.add(LeakyReLU(0.2))

    # we can add our fifth layer to compute more features without upsampling

    # here we add our fifth convolutional layer which uses feature maps (matrices) to be 
    # dot producted with our image to identify features,
    # here our feature maps are 4x4 and we output 128, we specify padding to be 
    # the same so the layer does not decrease the image dimensions
    model.add(Conv2D(128, 4, padding='same'))
    # here we add our activation function again
    model.add(LeakyReLU(0.2))

    # we now should have enough trainable params to identify features in our training
    # dataset and generate them (2.16 million params),
    # we can now add out final layer to get to a depth of 1

    # here we add our sixth convolutional layer which uses feature maps (matrices) to be 
    # dot producted with our image to identify features,
    # here our feature maps are 4x4 and we output 1, we specify padding to be 
    # the same so the layer does not decrease the image dimensions.
    # use a sigmoid activation as we want our pixels to be in the range of 0 and 1
    model.add(Conv2D(1, 4, padding='same', activation='sigmoid'))

    # we now have an output of 28x28x1 with enough convoloutional layers 
    # to identify most features
    return model

"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 6272)              809088    
                                                                 
 leaky_re_lu (LeakyReLU)     (None, 6272)              0         
                                                                 
 reshape (Reshape)           (None, 7, 7, 128)         0         
                                                                 
 up_sampling2d (UpSampling2D  (None, 14, 14, 128)      0         
 )                                                               
                                                                 
 conv2d (Conv2D)             (None, 14, 14, 128)       409728    
                                                                 
 leaky_re_lu_1 (LeakyReLU)   (None, 14, 14, 128)       0         
                                                                 
 up_sampling2d_1 (UpSampling  (None, 28, 28, 128)      0         
 2D)                                                             
                                                                 
 conv2d_1 (Conv2D)           (None, 28, 28, 128)       409728    
                                                                 
 leaky_re_lu_2 (LeakyReLU)   (None, 28, 28, 128)       0         
                                                                 
 conv2d_2 (Conv2D)           (None, 28, 28, 128)       262272    
                                                                 
 leaky_re_lu_3 (LeakyReLU)   (None, 28, 28, 128)       0         
                                                                 
 conv2d_3 (Conv2D)           (None, 28, 28, 128)       262272    
                                                                 
 leaky_re_lu_4 (LeakyReLU)   (None, 28, 28, 128)       0         
                                                                 
 conv2d_4 (Conv2D)           (None, 28, 28, 1)         2049      
                                                                 
=================================================================
Total params: 2,155,137
Trainable params: 2,155,137
Non-trainable params: 0
_________________________________________________________________
None
"""

# lets now build our discriminator
def build_discriminator():

    # we initialise our model
    model = Sequential()

    # we can create our first input layer

    # the first layer is a convolutional layer with an input of a 28x28 image
    # and an output of 32 with a 5x5 feature map. this layer with make our input
    # image smaller by 4 in each dimension
    model.add(Conv2D(32, 5, input_shape=(28,28,1)))
    # we can add our activation layer now
    model.add(LeakyReLU(0.2))
    #we can add a dropout to utilise more nuerons
    model.add(Dropout(0.4))

    # second layer is the same

    # the second layer is a convolutional layer with an input of a 28x28 image
    # and an output of 64 with a 5x5 feature map. this layer with make our input
    # image smaller by 4 in each dimension
    model.add(Conv2D(64, 5, input_shape=(28,28,1)))
    # we can add our activation layer now
    model.add(LeakyReLU(0.2))
    #we can add a dropout to utilise more nuerons
    model.add(Dropout(0.4))

    # third layer is the same

    # the second layer is a convolutional layer with an input of a 28x28 image
    # and an output of 128 with a 5x5 feature map. this layer with make our input
    # image smaller by 4 in each dimension
    model.add(Conv2D(128, 5, input_shape=(28,28,1)))
    # we can add our activation layer now
    model.add(LeakyReLU(0.2))
    #we can add a dropout to utilise more nuerons
    model.add(Dropout(0.4))

    # third layer is the same

    # the second layer is a convolutional layer with an input of a 28x28 image
    # and an output of 256 with a 5x5 feature map. this layer with make our input
    # image smaller by 4 in each dimension
    model.add(Conv2D(256, 5, input_shape=(28,28,1)))
    # we can add our activation layer now
    model.add(LeakyReLU(0.2))
    #we can add a dropout to utilise more nuerons
    model.add(Dropout(0.4))

    # we should now have enough layers to identify if the image is fake or not
    # lets flatten our ouput and now run it though a dense layer. we use sigmoid 
    # again to represent 1 as fake and 0 as real
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    return model

"""
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_5 (Conv2D)           (None, 24, 24, 32)        832       
                                                                 
 leaky_re_lu_5 (LeakyReLU)   (None, 24, 24, 32)        0         
                                                                 
 dropout (Dropout)           (None, 24, 24, 32)        0         
                                                                 
 conv2d_6 (Conv2D)           (None, 20, 20, 64)        51264     
                                                                 
 leaky_re_lu_6 (LeakyReLU)   (None, 20, 20, 64)        0         
                                                                 
 dropout_1 (Dropout)         (None, 20, 20, 64)        0         
                                                                 
 conv2d_7 (Conv2D)           (None, 16, 16, 128)       204928    
                                                                 
 leaky_re_lu_7 (LeakyReLU)   (None, 16, 16, 128)       0         
                                                                 
 dropout_2 (Dropout)         (None, 16, 16, 128)       0         
                                                                 
 conv2d_8 (Conv2D)           (None, 12, 12, 256)       819456    
                                                                 
 leaky_re_lu_8 (LeakyReLU)   (None, 12, 12, 256)       0         
                                                                 
 dropout_3 (Dropout)         (None, 12, 12, 256)       0         
                                                                 
 flatten (Flatten)           (None, 36864)             0         
                                                                 
 dropout_4 (Dropout)         (None, 36864)             0         
                                                                 
 dense_1 (Dense)             (None, 1)                 36865     
                                                                 
=================================================================
Total params: 1,113,345
Trainable params: 1,113,345
Non-trainable params: 0
_________________________________________________________________
None
"""

# lets build our models now
generator = build_generator()
discriminator = build_discriminator()


# our discriminator needs to learn slower as its task is easier 
# and we need balance in training.
g_opt = Adam(0.0001)
d_opt = Adam(0.00001)

g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()

# since we cannot just do model.fot as we want these models to train together
# we need to customize our own training loop to train the models side by side
class GAN_Fashion(Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        # if we want to use base model constructor
        super().__init__(*args, **kwargs)

        # initialising local gen and discrim
        self.generator = generator
        self.discriminator = discriminator

    
    # here we change our training for our gan model
    def train_step(self, batch):
        
        # getting the real image data
        real_images = batch
        # using a random input for our generator to generate a new image
        fake_images = self.generator(tf.random.normal((128, 128, 1)), training = False)

        # training the discriminator
        # we pass the real and fake images into our model and add some noise to slow it
        # down from learning to quickly, we then apply our loss and back propigate
        with tf.GradientTape() as d_tape:
            # here we passs in the real and fake images
            yhat_real = self.discriminator(real_images, training =  True)
            yhat_fake = self.discriminator(fake_images, training = True)
            # we now join our outputs to make one value
            yhat_real_fake  =tf.concat([yhat_real, yhat_fake], axis = 0)

            # we can now create our labels if they are real or fake
            y_real_fake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis = 0)

            # we can now add some noise to the outputs
            noise_real = 0.15 * tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15 * tf.random.uniform(tf.shape(yhat_fake))
            y_real_fake += tf.concat([noise_real, noise_fake], axis = 0)

            # we can now calc loss
            total_d_loss = self.d_loss(y_real_fake, yhat_real_fake)

        # now we can apply back propigation
        d_grad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))

        # lets train our generator now
        with tf.GradientTape() as g_tape:
            # generate new images
            gen_images = self.generator(tf.random.normal((128, 128, 1)), training = True)

            # creating predction labels
            pred_labels = self.discriminator(gen_images, training = False)

            # calculating loss using our discriminator
            total_g_loss = self.g_loss(tf.zeros_like(pred_labels), pred_labels)

        # applying back propigation
        g_grad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(g_grad, self.generator.trainable_variables))

        # returning our loss
        return {"d_loss": total_d_loss, "g_loss": total_g_loss}


    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs):
        # to use base comile if needed
        super().compile(*args, **kwargs)
        
        # defining our local loss and optimisers
        self.g_opt = g_opt
        self.g_loss = g_loss
        self.d_opt = d_opt
        self.d_loss = d_loss

# lets get an instance of this subclass
fashion_GAN = GAN_Fashion(generator, discriminator)

'''
# lets comile our model
fashion_GAN.compile(g_opt, d_opt, g_loss, d_loss)
'''
# we can store images from our generator as its training to see the progress
# we do this using callback
class ModelMonitor(Callback):
    # num_img is the number of images to generate, latent_dim is the input for the generator
    def __init__(self, num_img=3, latent_dim=128):
        # saving our passsed in variables
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        # getting our random values
        random_latent_vectors = tf.random.uniform((self.num_img, self.latent_dim,1))
        # running the values through our generator
        generated_images = self.model.generator(random_latent_vectors)
        # scaling back up our values
        generated_images *= 255
        # getting the ouput as a numpy array
        generated_images.numpy()
        # going through each generated array and turning it into an image and saving it
        for i in range(self.num_img):
            img = array_to_img(generated_images[i])
            # saved to images folder in path (YOU NEED TO CREATE THIS PATH)
            img.save(os.path.join('images', f'generated_img_{epoch}_{i}.png'))

# i already have pre trained weights i can load in, trained for 2000 epochs
'''
# finally we train our model (need 2000 epochs to make fully generated images like fashion)
# we also use our callback from before (optional)
losses = fashion_GAN.fit(ds, epochs=20, callbacks = [ModelMonitor()])
'''
# loading our saved weights
generator.load_weights('generatormodel-2.h5')

# lets see the result of the generator
# 16 is the number of images to generate then 128 is the input shape
imgs = generator.predict(tf.random.normal((16, 128, 1)))

fig, axes = plt.subplots(ncols = 4, nrows = 4, figsize=(20,20))

for r in range(4):
    for c in range(4):
        axes[r][c].imshow(imgs[(r+1)*(c+1)-1])

plt.show()