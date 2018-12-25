import os
import sys

# TensorFlow and tf.keras
import string
table = str.maketrans({key: None for key in string.punctuation})

import numpy as np
import tensorflow as tf
from tensorflow import keras
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import load_model

# Helper libraries
import matplotlib.pyplot as plt

print("ЛОГ: версия tensorflow:", tf.__version__)

imdb = keras.datasets.imdb
word_index = {k: (v + 3) for k, v in imdb.get_word_index().items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3
reverse_word_index = {value: key for key, value in word_index.items()}
print('ЛОГ: словарь инициализирован')


def encode_review(text: str):
    text = text.translate(table).lower()
    res = [1]
    for word in text.split(' '):
        el = word_index.get(word, 2)
        res.append(el if el < 10000 else 2)
    return np.array(res)


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


def compile_model(model):
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


def create_model(filename, epochs_count):
    (train_data, train_labels), _ = imdb.load_data(num_words=10000)

    # Дополняем данные до 256 символов (0-символом)
    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=word_index["<PAD>"],
                                                            padding='post',
                                                            maxlen=256)

    vocab_size = 10000
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
    print(model.summary())

    compile_model(model)

    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]

    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=epochs_count,
                        batch_size=512,
                        validation_data=(x_val, y_val),
                        verbose=1)

    history_dict = history.history
    history_dict.keys()

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, loss, 'bo', label='Ошибка обучения')
    plt.plot(epochs, val_loss, 'b', label='Ошибка проверки обучения')
    plt.title('Функция ошибки обучения и проверки при обучении')
    plt.xlabel('Итерация')
    plt.ylabel('Ошибка')
    plt.legend()

    plt.show()

    plt.clf()

    plt.plot(epochs, acc, 'bo', label='Корректность обучения')
    plt.plot(epochs, val_acc, 'b', label='Корректность проверки обучения')
    plt.title('Корректность обучения и проверки обучения')
    plt.xlabel('Итерация')
    plt.ylabel('Точность')
    plt.legend()

    plt.show()

    model.save(filename)


def test_model(model_filename):
    model = load_model(model_filename)
    compile_model(model)

    if sys.argv[3] == 'imdb':
        print('ЛОГ: тестирование сети с помощью базы imdb')
        _, (test_data, test_labels) = imdb.load_data(num_words=10000)
    else:
        print('ЛОГ: пользовательский тест')
        if not os.path.exists(sys.argv[3]):
            print('ОШИБКА: файл {} не существует'.format(sys.argv[3]))
            exit(1)
        with open(sys.argv[3]) as f:
            text = f.read()
        text = ' '.join(text.split('\n'))
        test_data = np.array([encode_review(text)])
        if len(test_data[0]) > 256:
            print('ПРЕДУПРЕЖДЕНИЕ: текст слишком длинный и будет обрезан. Максимальная длина 256 слов')
        test_labels = np.array([int(sys.argv[4])])

    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=word_index["<PAD>"],
                                                           padding='post',
                                                           maxlen=256)

    results = model.evaluate(test_data, test_labels)
    print('Точность = {:.2f}%'.format(results[1] * 100))


def main():
    op = sys.argv[1]
    if op == 'create':
        create_model(sys.argv[2], int(sys.argv[3]))
    if op == 'test':
        test_model(sys.argv[2])


if __name__ == '__main__':
    main()
