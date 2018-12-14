import sys
import json
from operator import add
from functools import reduce

SEPARATOR_LEN = 30
PRINT_DEBUG = '-v' in sys.argv or '-vv' in sys.argv
PRINT_DEBUG_FULL = '-vv' in sys.argv


def normalized(vec):
    d = reduce(add, map(lambda x: x ** 2, vec)) ** 0.5
    return [vi / d for vi in vec] if d > 0 else vec[:]


def float_str(f):
    return '{:.3f}'.format(f)


def float_arr_str(arr):
    return ', '.join(map(float_str, arr))


class NeuronNetworkParseError(ValueError):
    pass


class Printer:
    @staticmethod
    def neuron_computation(x, w, y):
        if PRINT_DEBUG_FULL:
            print_res = 'Результат нейрона: {}'.format(float_str(y))
            print_comp = ' + '.join(['{} * {}'.format(float_str(xi), float_str(wi)) for xi, wi in zip(x, w)])
            print('        {:<28} Подробно: ({})'.format(print_res, print_comp))
        elif PRINT_DEBUG:
            print('        Результат нейрона:', float_str(y))

    @staticmethod
    def layer_computation_start(x):
        if PRINT_DEBUG:
            print('    Вход слоя: ({})'.format(float_arr_str(x)))

    @staticmethod
    def layer_computation_end(y):
        if PRINT_DEBUG:
            print('    Результат слоя: ({})\n'.format(float_arr_str(y)))

    @staticmethod
    def net_computation_start(x):
        if PRINT_DEBUG:
            print('-' * SEPARATOR_LEN + ' ВЫЧИСЛЕНИЕ ' + '-' * SEPARATOR_LEN)
            print('Входной вектор x = ({})'.format(float_arr_str(x)), end='\n\n')

    @staticmethod
    def net_computation_end(y):
        if PRINT_DEBUG:
            print('Результат вычислений y = ({})'.format(float_arr_str(y)))
            print('-' * SEPARATOR_LEN + ' КОНЕЦ ВЫЧИСЛЕНИЙ ' + '-' * SEPARATOR_LEN)

    @staticmethod
    def train_25_rule(xy_len, w_count):
        if xy_len < 2 * w_count:
            print('ПРЕДУПРЕЖДЕНИЕ: нарушение правила 2-5: обучающая выбора слишком мала')
            print('ПРЕДУПРЕЖДЕНИЕ: минимальный объем выборки должен быть {}'.format(2 * w_count))
        elif 5 * w_count < xy_len:
            print('ПРЕДУПРЕЖДЕНИЕ: нарушение правила 2-5: обучающая выборка слишком велика')
            print('ПРЕДУПРЕЖДЕНИЕ: максимальный объем выборки должен быть {}'.format(5 * w_count))
        else:
            raise ValueError('Неверный вызов принта предупреждения нарушения правила 2-5')
        print('ПРЕДУПРЕЖДЕНИЕ: объем выборки: {}; количество параметров: {};'.format(xy_len, w_count))


def try_parse_int(line):
    try:
        v = float(line)
    except ValueError as e:
        raise NeuronNetworkParseError('Невозможно распознать строковое число {}'.format(line)) from e
    return v


def parse_line(ll):
    line_num, line = ll
    if not line.strip():
        return []
    numbers = line.replace(' ', '').split(',')
    try:
        numbers = list(map(try_parse_int, numbers))
    except NeuronNetworkParseError as e:
        raise NeuronNetworkParseError('Ошибка в строке данных {}'.format(line_num + 1)) from e
    return numbers


def parse_txt(filename):
    with open(filename) as f:
        lines = f.readlines()
    lines = list(map(parse_line, enumerate(lines)))

    idx = 0
    mats = []
    while idx < len(lines):
        mat = [lines[idx]]
        idx += 1
        while idx < len(lines) and lines[idx]:
            if len(lines[idx]) != len(mat[0]):
                raise NeuronNetworkParseError('Неверное число элементов матрицы в строке {}'.format(idx + 1))
            mat.append(lines[idx])
            idx += 1
        mats.append(mat)
        if idx == len(lines):
            raise NeuronNetworkParseError('Конец файла, но ожидалась следующая строка')
        idx += 1
    return mats


def parse_json(filename):
    with open(filename) as f:
        content = f.read()
    return json.loads(content)


def parse_x(filename):
    with open(filename) as f:
        line = f.read()
    try:
        x = list(map(try_parse_int, line.replace(' ', '').split(',')))
    except NeuronNetworkParseError as e:
        raise NeuronNetworkParseError('Ошибка в формате входного вектора') from e
    return x


def parse_xy(filename):
    with open(filename) as f:
        lines = f.readlines()

    xy = []
    for idx, line in enumerate(lines):
        if '->' not in line:
            raise NeuronNetworkParseError('Ошибка в формате обучающей выборки, строка {}'.format(idx))
        try:
            posl, posr = line.find('['), line.find(']')
            x = eval(line[posl:posr + 1])
            posl, posr = line.rfind('['), line.rfind(']')
            y = eval(line[posl: posr + 1])
            xy.append(([float(xi) for xi in x], [float(yi) for yi in y]))
        except ValueError as e:
            raise NeuronNetworkParseError('Ошибка в формате обучающей выборки, строка {}'.format(idx)) from e
    return xy
