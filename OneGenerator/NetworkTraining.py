from __future__ import print_function
from keras.models import Sequential
from keras import backend as K
from keras.optimizers import Adam
import numpy as np
import matplotlib
import time
matplotlib.use('Agg') # matplotlib likes to crash if you don't do this.
import matplotlib.pyplot as plt
from NetworkArchitectures import one_generator, one_discriminator
from keras.datasets.mnist import load_data

'''
Various Notes:
1) 1 is the discriminator's output for a "real" image, 0 is the output for a "synthetic" image.
2) The training code is very barebones - it has a fixed learning rate, and there's no real trick involved.
'''


def filter_digits(chosen_digit):
    (train_image, train_label), (test_image, test_label) = load_data()
    train_label = np.where(train_label == chosen_digit)
    test_label = np.where(test_label == chosen_digit)
    train = np.zeros([train_label[0].shape[0], 28, 28, 1])
    test = np.zeros([test_label[0].shape[0], 28, 28, 1])
    train[:,:,:,0] = train_image[train_label] / 255.0
    test[:,:,:,0] = test_image[test_label] / 255.0
    return train, test


def grab_discriminator_samples(dataset, samples, index, permutation, model):
    train_mixed_ones = np.zeros([samples,28,28,1])
    train_mixed_labels = np.zeros([samples])
    for j in range(0, samples // 2):
        train_mixed_ones[j, :, :, :] = dataset[permutation[(index*samples) + j], :, :, :]
        train_mixed_labels[j] = 1.0
    fakes = grab_generated_samples(samples//2, model)
    for j in range(samples//2, samples):
        train_mixed_ones[j, :, :, :] = fakes[j - samples//2, :, :, :]
        train_mixed_labels[j] = 0.0
    return train_mixed_ones, train_mixed_labels


def grab_generated_samples(samples, model):
    train_noise = 2 * np.random.random([samples, latent_dimension]) - 1
    synth_images = model.predict(train_noise)
    return synth_images


def make_gan(model, discriminator):
    discriminator.trainable = False  # Note: The discriminator being untrainable only applies to gan, not to the compiled discriminator above
    gan = Sequential()
    gan.add(model)
    gan.add(discriminator)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=learnrate), metrics=['accuracy'])
    return gan


# Load Models
latent_dimension = 32
model = one_generator(latent_dimension)
discriminator = one_discriminator()

# Set Training Parameters
chosen_digit = 1
batch_size = 8
epochs = 50
learnrate = 1e-4
label_noise_level = 0.00

# Load Data
train_ones, test_ones = filter_digits(chosen_digit)
train_size = train_ones.shape[0]
num_batches = train_size // batch_size

for i in range(25):
    plt.subplot(5,5,1+i)
    plt.axis('off')
    plt.imshow(train_ones[i,:,:,0], cmap='gray')
plt.savefig("images.png")

# Compile models.
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=learnrate), metrics=['accuracy'])

# Create GAN
gan = make_gan(model,discriminator)

# Train
train_loss_disc = np.zeros(epochs)
test_loss_disc = np.zeros(epochs)

for epoch in range(epochs):
    #Train generator and discriminator simultaneously
    ## Generate samples
    train_synth_ones = K.eval(model(K.random_uniform_variable((train_size,latent_dimension),-1,1)))
    permutation = np.random.permutation(train_size)
    for i in range(num_batches):
        start = time.time()
        print("Epoch " + str(epoch + 1) + "/" + str(epochs) + ": Batch " +str(i+1) +"/" +str(num_batches))
        train_mixed_ones, train_mixed_labels = grab_discriminator_samples(train_ones,batch_size,i,permutation, model)
        disc_mets = discriminator.train_on_batch(train_mixed_ones, train_mixed_labels)
        train_noise = 2 * np.random.random([batch_size, latent_dimension]) - 1
        train_gan_labels = np.ones(batch_size)
        model_mets = gan.train_on_batch(train_noise, train_gan_labels)
        time_taken = time.time() - start
        #print("Discriminator accuracy:" +str(disc_mets[1]) + "\t Generator Fool Rate:" + str(model_mets[1]) + "\t Estimated time per epoch:" + str(int(time_taken * num_batches)) + " seconds")
        print("Discriminator accuracy: %6.3f \t Generator Fool Rate: %6.3f \t Estimated time per epoch: %6.3f seconds" % (disc_mets[0],model_mets[0],(time_taken * num_batches)))

    ## Generate images for the given epoch

    synth_images = grab_generated_samples(25,model)
    disc_mets = discriminator.test_on_batch(synth_images, np.zeros([25]))
    print("Generator Fool Rate:" + str(disc_mets[0]))
    disc_mets = discriminator.test_on_batch(test_ones[range(0,25),:,:,:], np.zeros([25]))
    print("Discriminator True Pass Rate:" + str(disc_mets[0]))
    for i in range(25):
        plt.subplot(5,5,1+i)
        plt.axis('off')
        plt.imshow(synth_images[i,:,:,0], cmap='gray')
    plt.savefig("generated_images_epoch_" + str(epoch) + ".png")
