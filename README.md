# Image Generators

The goal of this set of projects is to build a neural network that produces various types of images. Each is built upon the last, small and incrementally.

## Sub-Projects Leading to the Major Project

### One Generator

**STATUS**: Complete

I have to start somewhere. The first project here is the "One Generator."

It was built with the goal of producing a picture of the number 1. It produces the desired output. Some improvements need to be made, but as a start it's somewhat reasonable.

### Four Generator

**Status**: Complete

This is the next logical step. This is designed to produce a picture of the number 4.

Why was this the next logical step? Because there are two reasonable ways to write the number 4. In some real sense, we are asking our network to produce one of two random results, and this could have reasonably caused mode collapse - that is, it might have only produced one of the types of results and never the second.

This was never a real problem for this architecture, and it didn't take much. 

### Digit Generator

**Status**: Planned

This is the final small step. This is designed to produce a picture of a desired digit.

I plan on using a Conditional GAN architecture to achieve this. Hopefully things go reasonably well here, and I can pick which digit I generate each time.

### Dataset Acquisition

**Status**: Unplanned - Considering Reasonable Datasets to Acquire

### Actual Art Generation

**Status**: Unplanned - Considering Reasonable Datasets to Acquire

