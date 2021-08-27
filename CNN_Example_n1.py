import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

print('Tensorflow version: ', tf.__version__)
print('Keras version: ', keras.__version__)

# Data download
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# Data preprocessing
train_images = train_images.reshape(60000, 28, 28, 1)
train_images = train_images.astype('float32')
train_images = train_images/255

test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images.astype('float32')
test_images = test_images/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels, num_classes=10)

model = keras.Sequential()
model.add(Conv2D(32, (5,5), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(32,(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=80, epochs=5, verbose=1)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy: ', test_acc)