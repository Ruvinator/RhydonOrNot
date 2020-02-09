# Rhydon... or not?
Is the Pokemon Rhydon in an image? I don't know, but let's use ML to find out!

### Data
There are plenty of images of Rhydon on the internet, but instead of scraping to build my dataset I decided to make it instead. Here's what I did:

1. Download 3D models of Rhydon and other cool Pokemon (Charizard, Gyarados, Blastoise)
2. Develop script in Unreal Engine 4 which loads and rotates around the 3D model at different viewing angles/positions, capturing screenshots
3. Change model pose and repeat step 2
4. Post-process screenshots (convert to grayscale to minimize data used by model)

### Model & Training
My model of choice was a convolutional neural network (CNN).

I developed helper functions for automatically splitting the images into training/validation/testing sets which were then fed to the CNN for training.

The model was saved after training was completed. I implemented functions for inferencing from the trained network so that any image can be input and the model generates an output (either Rhydon or not).

### Results
#### Testing accuracy: **99%**

#### Training accuracy: **100%**
