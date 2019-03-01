from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((train_images.shape[0], train_images.shape[1] * train_images.shape[2]))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((test_images.shape[0], test_images.shape[1] * test_images.shape[2]))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(train_images.shape[1],)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

network.fit(train_images, train_labels, epochs=5, batch_size=128, validation_split=0.1)

network.evaluate(test_images, test_labels)
