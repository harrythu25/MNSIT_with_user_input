import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)
mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']



num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples #0.1 because only want 10%
num_validation_samples = tf.cast(num_validation_samples, tf.int64) #cast to integer

num_test_samples = mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64) #cast to integer

#preparing the data sets before splitting

#take input and transform
def scale (image, label):
  image = tf.cast (image, tf.float32) #make sure it's float
  image /= 255. #values are between 0 and 255 based on the shades; divide everything by 255 so each input is between 0 and 1
  return image, label


#scale the data
scaled_train_and_validation_data = mnist_train.map(scale) #use function to transform train data
test_data = mnist_test.map(scale) #use function to transform test data

#shuffle data so batches wont affect model
BUFFER_SIZE = 10000

shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)

validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
train_data = shuffled_train_and_validation_data.skip(num_validation_samples)

#Because I am going to use the mini batch gradient descent, I need to set the batch size
#setting batch size to prepare data for model

BATCH_SIZE = 100

train_data = train_data.batch(BATCH_SIZE) #override .batch method with our size
validation_data = validation_data.batch(num_validation_samples)  #to take in the whole data set,
test_data = test_data.batch(num_test_samples)


validation_inputs, validation_targets = next(iter(validation_data))

input_size = 784 # 28px x 28px
output_size = 10 #0-9
hidden_layer_size = 100

model = tf.keras.Sequential([
                             tf.keras.layers.Flatten(input_shape = (28, 28, 1)),
                             tf.keras.layers.Dense(hidden_layer_size, activation= 'relu'), # first hidden layer
                             tf.keras.layers.Dense(hidden_layer_size, activation= 'relu'),  # second hidden layer
                             tf.keras.layers.Dense(output_size, activation='softmax') #output layer softmax to transform into probability since this is a classifier model
                            ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics= ['accuracy'])
#this aplies one hot encoding which I did not do
#output and target layer nees to have same shape of one hot encoded format


NUM_EPOCHS =5

model.fit(train_data, epochs=NUM_EPOCHS, validation_data=(validation_inputs, validation_targets), verbose =2 )

test_loss, test_accuracy = model.evaluate(test_data)
print('Test loss: {0: .2}. Test accuracy: {1: .2f}%'.format(test_loss, test_accuracy*100.))

model.save('model1.model')
print('Save successful')