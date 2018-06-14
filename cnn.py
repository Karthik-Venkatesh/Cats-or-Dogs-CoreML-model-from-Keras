# Cats or Dogs model
# Keras model to Apple CoreML model

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
# 2,2 is pooling matrix size
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
# output will be the one class that is class or dog. so output_dim = 1 and activation function is "sigmoid"
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
# adam is stocastic gradient descent algoritm name.
# The 'binary_crossentropy' is used for binary output cat or dog. If its multiple output 'categoriacl cross entropy'
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255) 

# Traget size (64,64) beacause the in Convolution step the is size was (64,64)
# class_mode = 'binary', Only 2 classes are there cats and dogs. 
# 32 images in one batch. 
training_set = train_datagen.flow_from_directory('dataset/training_set', target_size = (64, 64),  batch_size = 32, class_mode = 'binary')
test_set = test_datagen.flow_from_directory('dataset/test_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

# 'steps_per_epoch' is number of images in training set.
# 'validation_steps' is number of images in test set.
classifier.fit_generator(training_set, steps_per_epoch = 1900, epochs = 5,  validation_data = test_set, validation_steps = 100)

# Saving to CoreML Model
import coremltools
coreml_model = coremltools.converters.keras.convert(classifier)
coreml_model.save("cats_or_dogs.mlmodel")