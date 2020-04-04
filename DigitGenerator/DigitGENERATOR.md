# Digit Generator

The purpose of this neural network was to produce pictures of an arbitrary chosen digit. This should be the "ultimate" goal for this network with this data.

First, we wish to make a few comments. The developer's goal is to make the network as lightweight as possible, so that he can actually train the network on his computer. The goal is to run lightweight enough to be cheap to apply, and to be practical enough to train. Even so, this particular exercise requires something like a Conditional GAN.

The following is the architecture of the generator.

![Architecture of Digit Generator](Digit_Generator.png "Digit Generator Architecture")

As with the specific generators, we start with a random seed. Unlike the specific generators, however, we feed the "chosen digit" into a dense layer, which serves as a filter selector. The filters are then selected by multiplying the noise with the filter selector's output. We then have the network learn low-resolution features, followed by mid-resolution features, then high-resolution features.

The low, mid, and high resolution features are combined in the last concatenation steps and convolutional layer. The motivation is that we can understand roughly what the image is at low resolutions, learn improvements to it at higher resolutions, and combine the resulting images to make a good image.

The following is the architecture of the discriminator.

![Architecture of Digit Discriminator](Digit_Discriminator.png "Digit Discriminator Architecture")

We're given an image. We have the network learn high-resolution features, followed by mid-level features, then low-resolution features.

The low, mid, and high resolution features are then "scored" by taking the maximum across the channel - our MaxPool layers are designed to produce a single "score" for each channel. That is, it tells us how much of any given feature is present in the image.

These scores are then taken, concatenated together, and then fed into a dense layer to produce the classification.

This is the same discriminator as before, with one notable exception: instead of giving a score of "real or fake," the discriminator's goal is to classify the digit correctly if and only if it is a "real," and to spit out all zeroes if it is a "fake." This is notably different from the usual situation with a Conditional GAN, where the "condition" is just given to the discriminator. The hope here was that this could make the discriminator cheaper; it's now mostly just a strange MNIST classifier, and that's something that we can reasonably do.

The end result of training with the parameters in the code was the following set of digits.

![Digit Generator_Output](generated_images_epoch_99.png "Digit Generator Output")

This is an acceptable result for the time being. For reference, here is a collection of digits from the MNIST dataset used to train the network:

![Digit Generator_Training_Data](images.png "Digit Generator Training Images")

