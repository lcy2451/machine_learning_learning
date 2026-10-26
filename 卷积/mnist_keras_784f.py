from keras.datasets import mnist
from keras.models import  Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.np_utils import to_categorical

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# plt.imshow(X_train[0], cmap='gray')
# plt.show()

x_train = X_train.reshape(60000, 784) / 255.0
x_test = X_test.reshape(10000, 784) / 255.0

y_train = to_categorical(Y_train, 10)
y_test = to_categorical(Y_test, 10)

model = Sequential()
model.add(Dense(units=256, activation='relu', input_dim=784))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=SGD(learning_rate=0.05),metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5000, batch_size=4096)

loss, accuracy = model.evaluate(x_test, y_test)
print('loss:', loss)
print('accuracy:', accuracy)
