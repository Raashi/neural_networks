import sys
import json
import math
from operator import add
from functools import reduce


ACTIVATION_ALPHA = 0.2


def func_activate(x):
    return 1 / (1 + math.exp(-ACTIVATION_ALPHA * x))


class Neuron:
    def __init__(self, weights):
        self.ins = len(weights)
        self.weights = weights

    def compute(self, x):
        res = func_activate(reduce(add, map(lambda xw: xw[0] * xw[1], zip(x, self.weights))))
        print('\t\tРезультат нейрона:', res)
        return res


class Layer:
    def __init__(self, mat):
        self.ins = len(mat)
        self.outs = len(mat[0])
        self.neurons = [Neuron(list(map(lambda idx: mat[idx][neur], range(len(mat))))) for neur in range(len(mat[0]))]

    def compute(self, x):
        res = [neuron.compute(x) for neuron in self.neurons]
        print('\tРезультат слоя:', res)
        return res


class Network:
    def __init__(self, mats):
        if isinstance(mats, dict):
            self.from_json(mats)
        elif isinstance(mats, list):
            self.layers = [Layer(mat) for mat in mats]
            self.ins = len(mats[0])
            self.outs = len(mats[-1][0])
        else:
            raise ValueError('Wrong mats parameter type')

    def compute(self, x):
        print('Входной вектор:', x)
        if len(x) != self.ins:
            raise ValueError('Неверная размерность входного вектора. Должна быть {}'.format(self.ins))
        for layer in self.layers:
            x = layer.compute(x)
        return x

    def to_json(self):
        obj = {
            'ins': self.ins,
            'outs': self.outs,
            'layers': []
        }
        for layer in self.layers:
            layer_obj = {
                'ins': layer.ins,
                'outs': layer.outs,
                'neurons': []
            }
            obj['layers'].append(layer_obj)
            for neuron in layer.neurons:
                layer_obj['neurons'].append({
                    'ins': neuron.ins,
                    'weights': neuron.weights
                })
        return obj

    def from_json(self, obj):
        self.ins = obj['ins']
        self.outs = obj['outs']
        self.layers = []
        for layer_obj in obj['layers']:
            mat = [[0] * layer_obj['outs'] for _i in range(layer_obj['ins'])]
            for neuron_idx, neuron_obj in enumerate(layer_obj['neurons']):
                for idx, w in enumerate(neuron_obj['weights']):
                    mat[idx][neuron_idx] = w
            self.layers.append(Layer(mat))


def parse_line(line):
    return list(map(int, line.replace(' ', '').split(',')))


def parse_txt(filename):
    with open(filename) as f:
        lines = f.readlines()
    lines = list(map(parse_line, lines))

    idx = 0
    mats = []
    while idx < len(lines) and lines[idx]:
        mat = [lines[idx]]
        idx += 1
        while idx < len(lines) and len(lines[idx]) == len(mat[0]):
            mat.append(lines[idx])
            idx += 1
        mats.append(mat)
    return mats


def parse_json(filename):
    with open(filename) as f:
        content = f.read()
    return json.loads(content)


def parse_x(filename):
    with open(filename) as f:
        line = f.read()
    return list(map(float, line.replace(' ', '').split(',')))


def main():
    if sys.argv[1] == '-i':
        net = Network(parse_txt(sys.argv[2]))
        with open('nn.json', 'w') as f:
            json.dump(net.to_json(), f, indent=4)
    elif sys.argv[1] == '-c':
        net = Network(parse_json(sys.argv[2]))
        y = net.compute(parse_x(sys.argv[3]))
        print('Результат вычислений y =', y)


if __name__ == '__main__':
    main()
