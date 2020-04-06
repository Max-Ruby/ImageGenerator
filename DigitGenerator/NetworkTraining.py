from __future__ import print_function
from keras.models import Sequential
from keras import backend as K
from keras.optimizers import Adam
from keras.losses import kullback_leibler_divergence, mean_absolute_error
import numpy as np
import matplotlib
import time
matplotlib.use('Agg') # matplotlib likes to crash if you don't do this.
import matplotlib.pyplot as plt
from NetworkArchitectures import digit_generator, digit_discriminator, make_gan
from keras.datasets.mnist import load_data
import tensorflow as tf
'''
Various Notes:
1) 1 is the discriminator's output for a "real" image, 0 is the output for a "synthetic" image.
2) The training code is very barebones - it has a fixed learning rate, and there's no real trick involved.
'''


def grab_discriminator_samples(dataset, labels, samples, index, permutation, model):
    train_mixed_digits = np.zeros([samples,28,28,1])
    train_mixed_labels = np.zeros([samples,11])
    for j in range(0, samples):
        train_mixed_digits[j, :, :, :] = dataset[permutation[(index*samples) + j], :, :, :]
        for k in range(0,10):
            train_mixed_labels[j,k] = labels[permutation[(index*samples)+j],k]
    return train_mixed_digits, train_mixed_labels


def grab_discriminator_real_samples(dataset, labels, samples, index, permutation):
    train_mixed_digits = np.zeros([samples,28,28,1])
    train_mixed_labels = np.zeros([samples,11])
    for j in range(0, samples// 2):
        train_mixed_digits[j, :, :, :] = dataset[permutation[(index*samples) + j], :, :, :]
        for k in range(0,10):
            train_mixed_labels[j,k] = labels[permutation[(index*samples)+j],k]
    fakes = grab_generated_samples(samples//2, model)
    for j in range(samples//2, samples):
        train_mixed_digits[j, :, :, :] = fakes[j - samples//2, :, :, :]
        train_mixed_labels[j,:] = np.zeros([11])
        train_mixed_labels[j,10] = 1
    return train_mixed_digits, train_mixed_labels


def grab_discriminator_fake_samples(samples, model):
    train_mixed_digits = np.zeros([samples,28,28,1])
    train_mixed_labels = np.zeros([samples,11])
    fakes = grab_generated_samples(samples//2, model)
    for j in range(0, samples):
        train_mixed_digits[j, :, :, :] = fakes[j - samples//2, :, :, :]
        train_mixed_labels[j,:] = np.zeros([11])
        train_mixed_labels[j,10] = 1
    return train_mixed_digits, train_mixed_labels

def embedding(val):
    emb = np.zeros([10])
    emb[val] = 1
    return emb


def grab_generated_samples(samples, model):
    train_noise = 2 * np.random.random([samples, latent_dimension]) - 1
    train_digits = np.zeros([samples,10])
    for i in range(0,samples):
        train_digits[i,:] = embedding(np.random.randint(0,9))
    synth_images = model.predict([train_noise,train_digits])
    return synth_images

# Load Models
latent_dimension = 256
model = digit_generator(latent_dimension,10)
discriminator = digit_discriminator(10)

# Set Training Parameters
batch_size = 32
epochs = 100
learnrate = 5e-4
learnrate_disc = 5e-5
label_noise_level = 0.00

# Load Data
(train_images_raw, train_labels_raw), (test_images_raw, test_labels_raw) = load_data()
train_size = train_labels_raw.shape[0]
train_images = np.zeros([train_size,28,28,1])
train_labels = np.zeros([train_size,10])
for i in range(0,train_size):
    train_images[i,:,:,0] = train_images_raw[i,:,:]/255.0
    train_labels[i,:] = embedding(train_labels_raw[i])
num_batches = train_size // batch_size


# Show off a few basic examples (Comment code out once run once, for efficiency.)
'''
for i in range(10):
    for j in range(10):
        plt.subplot(10,10,1+10*i+j)
        plt.axis('off')
        example_digits = train_images[train_labels_raw == i]
        plt.imshow(example_digits[j,:,:,0], cmap='gray')
plt.savefig("images.png")
'''

# Compile models.
discriminator.compile(loss='categorical_crossentropy',optimizer=Adam(lr=learnrate_disc, decay=1e-6),metrics=['accuracy'])

# Create GAN
gan = make_gan(model,discriminator,latent_dimension,10)

gan.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learnrate, decay=1e-6),metrics=['accuracy'])
# Train
train_loss_disc = np.zeros(epochs)
test_loss_disc = np.zeros(epochs)

for epoch in range(epochs):
    #Train generator and discriminator simultaneously

    permutation = np.random.permutation(train_size)
    for i in range(num_batches):
        start = time.time()
        print("Epoch " + str(epoch + 1) + "/" + str(epochs) + ": Batch " +str(i+1) +"/" +str(num_batches))
        # Grab Real samples and Synthetic samples, for discriminator training
#        train_mixed_digits, train_mixed_labels = grab_discriminator_samples(train_images,train_labels,batch_size,i,permutation, model)
        train_fake_digits, train_fake_labels = grab_discriminator_fake_samples(batch_size//2,model)
        train_real_digits, train_real_labels = grab_discriminator_real_samples(train_images,train_labels,batch_size//2,i,permutation)
        #disc_mets = discriminator.train_on_batch(train_mixed_digits, train_mixed_labels)
        disc_mets_1 = discriminator.train_on_batch(train_fake_digits,train_fake_labels)
        disc_mets_2 = discriminator.train_on_batch(train_real_digits,train_real_labels)
        train_noise = 2 * np.random.random([batch_size, latent_dimension]) - 1
        digits = np.zeros([batch_size, 10])
        target_fool = np.zeros([batch_size,11])
        for j in range(0, batch_size):
            digits[j, :] = embedding(np.random.randint(0, 9))
            for k in range(0,10):
                target_fool[j,k] = digits[j,k]
        model_mets = gan.train_on_batch([train_noise,digits], target_fool)
        time_taken = time.time() - start
        print("Discriminator accuracy:" +str(disc_mets_1[1]) + "\t Generator Fool Rate:" + str(model_mets[1]) + "\t Estimated time per epoch:" + str(int(time_taken * num_batches)) + " seconds")

        print("Discriminator loss:" + '%.5f' % disc_mets_1[0] + "\t Generator loss:" + str(model_mets[0]))

        print("Discriminator classification accuracy:" + str(disc_mets_2[1]) + "\t Discriminator classification loss:" + str(disc_mets_2[0]))
    ## Generate images for the given epoch
    for i in range(10):
        noise = 2 * np.random.random([10, latent_dimension]) - 1
        digit = np.zeros([10,10])
        for j in range(10):
            digit[j,:] = embedding(i)
        example_images = model.predict([noise,digit])
        for j in range(10):
            plt.subplot(10, 10, 1 + 10 * i + j)
            plt.axis('off')
            plt.imshow(example_images[j, :, :, 0], cmap='gray')
    plt.savefig("generated_images_epoch_" + str(epoch) + ".png")
