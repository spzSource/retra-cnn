from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Convolution2D, AveragePooling2D, Dense, Flatten

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

model = Sequential([
    Convolution2D(8, 3, 3, input_shape=(3, 32, 32), activation='relu'),
    Convolution2D(16, 5, 5, activation='relu'),
    AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='same'),
    Flatten(),
    Dense(256),
    Dense(10, activation='softmax')
])

model.compile(
    loss='mse',
    metrics=['accuracy'],
    optimizer=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False))


def to_possibility_array(class_number):
    result = []
    for i in range(0, 10):
        if i == class_number:
            result.append(1)
        else:
            result.append(0)
    return result


model.fit(X_train, map(to_possibility_array, y_train), nb_epoch=100, batch_size=256)
model.save('model.mdl')
