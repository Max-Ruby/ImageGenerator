# Four Generator

The purpose of this neural network was to produce pictures of fours. A four should be the hardest digit to draw, and if we can use our network to draw the number 4, then we should feel confident in our general approach.

The following is the architecture of the generator.

![Architecture of Four Generator](Four_Generator.png "Four Generator Architecture")

As with the one generator, we start with a random seed. We then have the network learn low-resolution features, followed by mid-resolution features, then high-resolution features.

The low, mid, and high resolution features are combined in the last concatenation steps and convolutional layer. The motivation is that we can understand roughly what the image is at low resolutions, learn improvements to it at higher resolutions, and combine the resulting images to make a good image.

The following is the architecture of the discriminator.

![Architecture of Four Discriminator](Four_Discriminator.png "Four Discriminator Architecture")

We're given an image. We have the network learn high-resolution features, followed by mid-level features, then low-resolution features.

The low, mid, and high resolution features are then "scored" by taking the maximum across the channel - our MaxPool layers are designed to produce a single "score" for each channel. That is, it tells us how much of any given feature is present in the image.

These scores are then taken, concatenated together, and then fed into a dense layer to produce the classification.

The end result of training with the parameters in the code was the following set of fours.

![Four Generator_Output](generated_images_epoch_99.png "Four Generator Output")

This is an acceptable result for the time being. For reference, here is a collection of fours from the MNIST dataset used to train the network:

![Four Generator_Training_Data](images.png "Four Generator Training Images")

