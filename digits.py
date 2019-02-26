from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images[0][0][0])
print(train_images.ndim)
print(train_images.shape)


