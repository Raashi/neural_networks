from operator import add
from functools import reduce

from utils import *


ACTIVATION_ALPHA = Decimal(5)


def func_activate(x):
    return Decimal(2) / (Decimal(1) + (-ACTIVATION_ALPHA * x).exp()) - Decimal(1)


class Neuron:
    def __init__(self, weights):
        self.ins = len(weights)
        self.weights = weights

    def compute(self, x):
        y = func_activate(reduce(add, map(lambda xw: xw[0] * xw[1], zip(x, self.weights))))
        Printer.neuron_computation(x, self.weights, y)
        return y


class Layer:
    def __init__(self, mat):
        self.ins = len(mat)
        self.outs = len(mat[0])
        self.neurons = [Neuron(list(map(lambda idx: mat[idx][neur], range(len(mat))))) for neur in range(len(mat[0]))]

    def compute(self, x):
        Printer.layer_computation_start(x)
        y = [neuron.compute(x) for neuron in self.neurons]
        Printer.layer_computation_end(y)
        return y


class Network:
    def __init__(self, mats):
        if isinstance(mats, dict):
            self.from_json(mats)
        elif isinstance(mats, list):
            self.layers = [Layer(mat) for mat in mats]
            self.ins = len(mats[0])
            self.outs = len(mats[-1][0])
        else:
            raise TypeError('Неверный тип аргумента в конструкторе Network')

    def compute(self, x):
        Printer.net_computation_start(x)
        if len(x) != self.ins:
            raise ValueError('Неверная размерность входного вектора. Должна быть {}'.format(self.ins))
        for idx, layer in enumerate(self.layers):
            x = layer.compute(x)
        Printer.net_computation_end(x)
        return x

    def train(self, xy):
        if len(xy[0][0]) != self.ins:
            raise ValueError()

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
                    mat[idx][neuron_idx] = Decimal(str(w))
            self.layers.append(Layer(mat))


def main():
    if sys.argv[1] == '-i':
        net = Network(parse_txt(sys.argv[2]))
        with open('nn.json', 'w') as f:
            json.dump(net.to_json(), f, indent=4)
    elif sys.argv[1] == '-c':
        x = parse_x(sys.argv[3])
        print('Входной вектор x = ({})'.format(arr_str_decimal(x)))
        net = Network(parse_json(sys.argv[2]))
        y = net.compute(x)
        print('Результат вычислений y = ({})'.format(arr_str_decimal(y)))


if __name__ == '__main__':
    main()
