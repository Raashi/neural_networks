import sys
import json
from decimal import getcontext, Decimal

PRECISION = 3
getcontext().prec = PRECISION

SEPARATOR_LEN = 30
PRINT_DEBUG = '-v' in sys.argv or '-vv' in sys.argv
PRINT_DEBUG_FULL = '-vv' in sys.argv


class NeuronNetworkParseError(ValueError):
    pass


def str_decimal(dec):
    return str((float(dec)))


def arr_str_decimal(arr_dec):
    return ', '.join(map(str_decimal, arr_dec))


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
            print('-' * SEPARATOR_LEN + ' ВЫЧИСЛЕНИЕ ' + '-' * SEPARATOR_LEN, end='\n\n')

    @staticmethod
    def net_computation_end(y):
        if PRINT_DEBUG:
            print('-' * SEPARATOR_LEN + ' КОНЕЦ ВЫЧИСЛЕНИЙ ' + '-' * SEPARATOR_LEN)


def parse_line(line):
    return list(map(int, line.replace(' ', '').split(','))) if line.strip() else []


def parse_txt(filename):
    with open(filename) as f:
        lines = f.readlines()
    lines = list(map(parse_line, lines))

    idx = 0
    mats = []
    while idx < len(lines):
        mat = [lines[idx]]
        idx += 1
        while lines[idx]:
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
    return list(map(Decimal, line.replace(' ', '').split(',')))