from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from keras.optimizers import SGD


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1) / 255.0
X_test = X_test.reshape(10000, 28, 28, 1) / 255.0

Y_train = to_categorical(y_train, 10)
Y_test = to_categorical(y_test, 10)

model = Sequential()
model.add(
    Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), padding='valid', input_shape=(28, 28, 1), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# 送入训练
model.compile(loss='categorical_crossentropy',optimizer=SGD(learning_rate=0.05),metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=5000, batch_size=4096)

# 评估测试集
loss, accuracy = model.evaluate(X_test, Y_test)
print('loss:', loss)
print('accuracy:', accuracy)
