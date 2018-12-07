import sys
import random
from utils import Decimal, arr_str_decimal

DEFAULT_LAYER_COUNT = 3
DEFAULT_LAYER_NEURON_COUNT_MIN = 2
DEFAULT_LAYER_NEURON_COUNT_MAX = 5
DEFAULT_X_MIN = 1
DEFAULT_X_MAX = 100

X_MIN = -100
X_MAX = 100

W_MIN = -3
W_MAX = 3


def get_random():
    return Decimal(random.uniform(X_MIN, X_MAX))


def get_random_weight():
    return random.randint(W_MIN, W_MAX)


def func_1(count):
    xy = []
    for _idx in range(count):
        x = [get_random(), get_random()]
        y = [Decimal(1) if x[0] > 0 and x[1] > 0 else
             Decimal(0.5) if x[0] < 0 < x[1] else
             Decimal(-0.5) if x[0] < 0 and x[1] < 0 else
             Decimal(-1)]
        xy.append((x, y))
    for _idx in range(count // 16):
        xy.append(([get_random(), 0], [0]))
    for _idx in range(count // 16):
        xy.append(([0, get_random()], [0]))
    xy.append(([0, 0], [0]))
    random.shuffle(xy)
    return xy


def func_2(count):
    xy = []
    for _idx in range(count):
        x = [random.uniform(0, 10)]
        y = [1 if x[0] > 5 else -1]
        xy.append((x, y))
    return xy


def gen_net():
    if '-l' in sys.argv:
        layers_sizes = list(map(int, sys.argv[sys.argv.index('-l') + 1:]))
        x_size = layers_sizes[0]
        layers_sizes = layers_sizes[1:]
        layer_count = len(layers_sizes)
    else:
        layer_count = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LAYER_COUNT
        neuron_min = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_LAYER_NEURON_COUNT_MIN
        neuron_max = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_LAYER_NEURON_COUNT_MAX

        print('Нейронная сеть будет иметь {} слоев'.format(layer_count))
        layers_sizes = [random.randint(neuron_min, neuron_max) for _idx in range(layer_count)]
        x_size = random.randint(neuron_min, neuron_max)

    print('Количество нейронов в слоях (от 1 до {})'.format(layer_count))
    print(str(layers_sizes)[1:-1])
    print('Размерность входного вектора равна {}'.format(x_size))
    y_size = layers_sizes[-1]
    print('Размерность выходного вектора равна {}'.format(y_size))

    mats = []
    for last_layer_size, next_layer_size in zip([x_size] + layers_sizes[:-1], layers_sizes):
        mat = [[get_random_weight() for _j in range(next_layer_size)] for _i in range(last_layer_size)]
        mats.append(mat)
    print('Матрицы сгенерированы')

    with open('nn.txt', 'w') as f:
        for mat in mats:
            for row in mat:
                f.write(', '.join(map(str, row)) + '\n')
            f.write('\n')


def main():
    if '-xy' in sys.argv:
        func_name = sys.argv[2]
        count = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        filename = sys.argv[4] if len(sys.argv) > 4 else 'xy.txt'
        xy = globals()['func_' + func_name](count)
        with open(filename, 'w') as f:
            for xi, yi in xy:
                f.write('[{}] -> [{}]\n'.format(arr_str_decimal(xi), arr_str_decimal(yi)))
    else:
        gen_net()


if __name__ == '__main__':
    main()
