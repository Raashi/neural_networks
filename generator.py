import sys
import random

DEFAULT_LAYER_COUNT = 3
DEFAULT_LAYER_NEURON_COUNT_MIN = 2
DEFAULT_LAYER_NEURON_COUNT_MAX = 5
DEFAULT_X_MIN = 1
DEFAULT_X_MAX = 100


def generate(layer_count, neuron_min, neuron_max):
    print('Нейронная сеть будет иметь {} слоев'.format(layer_count))

    layers_sizes = [random.randint(neuron_min, neuron_max) for _idx in range(layer_count)]
    print('Количество нейронов в слоях (от 1 до {})'.format(layer_count))
    print(str(layers_sizes)[1:-1])

    x_size = random.randint(neuron_min, neuron_max)
    print('Размерность входного вектора равна {}'.format(x_size))
    y_size = layers_sizes[-1]
    print('Размерность выходного вектора равна {}'.format(y_size))

    mats = []
    for last_layer_size, next_layer_size in zip([x_size] + layers_sizes[:-1], layers_sizes):
        mat = [[0 for _j in range(next_layer_size)] for _i in range(last_layer_size)]

        for neuron_next in range(next_layer_size):
            weights = [neuron + 1 for neuron in range(last_layer_size)]
            random.shuffle(weights)

            for last_layer_neuron, weight in enumerate(weights):
                mat[last_layer_neuron][neuron_next] = weight

        mats.append(mat)
    print('Матрицы сгенерированы')

    with open('tests/nn.txt', 'w') as f:
        for mat in mats:
            for row in mat:
                f.write(', '.join(map(str, row)) + '\n')

    x = [random.randint(DEFAULT_X_MIN, DEFAULT_X_MAX) for _idx in range(x_size)]
    print('Входной вектор:', ', '.join(map(str, x)))
    with open('tests/x.txt', 'w') as f:
        f.write(', '.join(map(str, x)))


def main():
    layer_count = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LAYER_COUNT
    neuron_min = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_LAYER_NEURON_COUNT_MIN
    neuron_max = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_LAYER_NEURON_COUNT_MAX
    generate(layer_count, neuron_min, neuron_max)


if __name__ == '__main__':
    main()
