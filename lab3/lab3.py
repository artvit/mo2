import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import dataloader

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer, AveragePooling2D, Dropout

dataset = dataloader.dataset_large
rnd_seed = 42


def init_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=6, kernel_size=5, activation='relu'))
    # model.add(Conv2D(filters=16, kernel_size=6, activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Flatten())
    model.add(Dense(300, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    print(model.summary())
    return model


def init_lenet5():
    model = Sequential()
    model.add(InputLayer(input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=6, kernel_size=3, activation='relu', padding='same'))
    # model.add(Dropout(rate=0.8))
    model.add(AveragePooling2D())
    model.add(Conv2D(filters=16, kernel_size=3, activation='relu'))
    # model.add(Dropout(rate=0.8))
    model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    print(model.summary())
    return model


def load_data():
    data, results = dataloader.load_data(dataset)
    results = tf.keras.utils.to_categorical(results)
    data.shape = (-1, 28, 28, 1)

    x_train, x_test, y_train, y_test = train_test_split(data, results, test_size=0.1, random_state=rnd_seed)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=rnd_seed)

    return x_train, y_train, x_test, y_test, x_valid, y_valid


def main():
    tr_x, tr_y, te_x, te_y, v_x, v_y = load_data()
    # model = init_model()
    model = init_lenet5()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(tr_x, tr_y, batch_size=16, epochs=10, validation_data=(v_x, v_y))
    score, acc = model.evaluate(te_x, te_y,
                                batch_size=16)
    print('Test score:', score)
    print('Test accuracy:', acc)


if __name__ == '__main__':
    main()
