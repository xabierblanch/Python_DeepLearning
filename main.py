import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

print('Tensorflow version: ', tf.__version__)
print('Keras version: ', keras.__version__)

# Data download
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Data preprocessing
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train/255
x_test = x_test/255

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Model definition

model = keras.Sequential()
model.add(Dense())
