# Side-Face-to-Front-Face-Conversion-using-Pix2Pix-Gan


Through this project we are trying to improve the effectiveness of face recognition systems. Current
face recognition systems do not give desired accuracy when the only input they are provided with
are side view images of the faces. So a person not directly facing the camera poses a problem for
these face recognition systems. To deal with this problem, we aim to use the side view and generate
a frontal image of the face of the same person and use it identify it.
Our algorithm has used Haar Cascade classifier provided by OpenCV to detect and extract the face
region of any image. But since Haar Cascade algorithms work for front faces and left facing images,
we deal with this problem a couple of ways – First we try to flip the image from right to left using
OpenCV functions and then try to detect the face region. If the result is still inconclusive, we run it
through a deep learning model created and trained in Caffe. For this purpose we have used a
pretrained model whose architecture has been stored into a text file and trained weights and
parameters have been stored in a .caffemodel file. Through these measures, we have been able to
increase the accuracy of our system from around 70% to near about 95%.
If the image of thecropped face is detected to be a side face – then we are giving that image as input
to out next module – a deep neural network with a GAN like architecture through which the source
image of a side view face is converted into it’s front face counterpart.
Our algorithm has been trained on similarly cropped face images of 95 pairs of side and front images
of the same person. For training we were earlier using a dataset of 100 foreign people of different
ethnicities but the model was then adapting to that dataset a lot such that the facial features of the
people used in training were then appearing in the faces generated for our test images ( images of
our own faces ). So we decided to train our model on Indian faces – so we asked help from our own
friends and classmates who agreed for a photoshoot. We took on an average 5 photos per person –
one front, right side 90 degree, right side 45 degree, left 90, left 45. We did this for 24 different
people.
To prepare our dataset we created image pairs – a side view face that served as the source image
and the front face of the same person which served as the target image. All images were of cropped
faces and were 256 * 256 RGB pixels large.
Like a normal GAN architecture, our model architecture consists of two different models – generator
and a discriminator. Generator is given the source image as input. It runs about seven layers of
convolutions ( conv2d layers paired with batch normalization and LeakyReLu activation ). Through
these seven layers, it runs about 512 image filters over the image, getting 512 different outputs –
one per filter – each output with a trained important feature of the image. In our case, the size of
the final 512 output is 1*1 pixels – i.e. at the end of the encoder part of the generator model, we
have 512 most important features/ pixels ( reduced from 256 * 256 = several thousand pixels ). You
can also see this as us compressing the image into a smaller feature set or extracting the most
important/ principal components of the images.
Note that number of filters in our layers were increased from 32 -&gt; 64 -&gt; 128 -&gt;256 -&gt; 512 as we
moved through the layers.
Once the encoding is done, generator uses the decoder layer blocks - seven layers of
conv2dtranspose – reverse of convolution again paired with LeakyReLu and BatchNormalization. This

time as we move through the layers we are decresing the filter numbers from 512 -&gt; 256-&gt; 128 -&gt; 64
-&gt; 32. At the end of this, we again have an image of 256*256 RGB pixels.
This block has guessed the missing values ( values lost during encoding ) and created an image that
tries to mimic the real image.
Once the image has been generated , it is sent to the Discriminator – this model tries to distinguish if
the image given to it is a real image or a fake image generate by the generator model.
These two models compete to fool each other – generator tries to create an image that
discriminator can not distinguish while the discriminator tries to identify the fake image sof the
generator. This competition between the two models drives the training model with the models
adjusting their weights after each victory or defeat.
In regular GAN models, the training is not very specific i.e when we give images of faces, GAN learns
to create faces but the faces generated are random. The GAN model does not try to target a specific
face which is what we aim to do through this algorithms. That is why we had to tweak this model a
bit – to make it work on the target condition we aim to provide it i.e the condition that hte face
generated must match a certain individual – the target.
So this is where the target image is used. Our discriminator model is not only given the generated
image to detect if it’s real or not and calculate binary cross entropy loss accordingly to use to train
the model, but is also given the target image. Then the discriminator calculate a pixel by pixel mean
absolute error – which is then used to guide the model into learning to create not a random face but
the front face of the target person.
Once the models are trained, we use the predict function of the generator by giving it our side face
image and it uses it’s trained weights to create the new front image. This image can then be run
throught he image recognition algorithms.
Things to note – One of the adjustments we had to do was to normalize the pixel values between -1
to 1 instead of 0 to 1 as normal image algorithms do to get better training. Without it, the generated
images were coming blank. Since our pixel values are negative sometimes, we use LeakyReLu instead
of simple ReLu to make sure the negative gradient is learned ( since ReLu has no slope for negative
value unlike LeakyReLu).
