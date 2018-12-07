import sys
import json
from decimal import getcontext, Decimal, InvalidOperation

PRECISION = 3
getcontext().prec = PRECISION

SEPARATOR_LEN = 30
PRINT_DEBUG = '-v' in sys.argv or '-vv' in sys.argv
PRINT_DEBUG_FULL = '-vv' in sys.argv


def create_decimal(arg):
    res = Decimal_Orig(arg) * Decimal_Orig(1)
    return float(arg)


Decimal_Orig = Decimal
Decimal = create_decimal


class NeuronNetworkParseError(ValueError):
    pass


def arr_str_decimal(arr_dec):
    return ', '.join(map(str, arr_dec))


class Printer:
    @staticmethod
    def neuron_computation(x, w, y):
        if PRINT_DEBUG_FULL:
            print_res = 'Результат нейрона: {}'.format(y)
            print_comp = ' + '.join(['{} * {}'.format(xi, wi) for xi, wi in zip(x, w)])
            print('        {:<28} Подробно: ({})'.format(print_res, print_comp))
        elif PRINT_DEBUG:
            print('        Результат нейрона:', y)

    @staticmethod
    def layer_computation_start(x):
        if PRINT_DEBUG:
            print('    Вход слоя: ({})'.format(arr_str_decimal(x)))

    @staticmethod
    def layer_computation_end(y):
        if PRINT_DEBUG:
            print('    Результат слоя: ({})\n'.format(arr_str_decimal(y)))

    @staticmethod
    def net_computation_start(x):
        if PRINT_DEBUG:
            print('-' * SEPARATOR_LEN + ' ВЫЧИСЛЕНИЕ ' + '-' * SEPARATOR_LEN)
            print('Входной вектор x = ({})'.format(arr_str_decimal(x)), end='\n\n')

    @staticmethod
    def net_computation_end(y):
        if PRINT_DEBUG:
            print('Результат вычислений y = ({})'.format(arr_str_decimal(y)))
            print('-' * SEPARATOR_LEN + ' КОНЕЦ ВЫЧИСЛЕНИЙ ' + '-' * SEPARATOR_LEN)


def try_parse_int(line):
    try:
        v = Decimal(line)
    except InvalidOperation:
        raise NeuronNetworkParseError('Невозможно распознать строковое число {}'.format(line))
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
        while lines[idx]:
            if len(lines[idx]) != len(mat[0]):
                raise NeuronNetworkParseError('Неверное число элементов матрицы в строке {}'.format(idx + 1))
            mat.append(lines[idx])
            idx += 1
        mats.append(mat)
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
            xy.append(([Decimal(xi) for xi in x], [Decimal(yi) for yi in y]))
        except InvalidOperation as e:
            raise NeuronNetworkParseError('Ошибка в формате обучающей выборки, строка {}'.format(idx)) from e

    return xy
