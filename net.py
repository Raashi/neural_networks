import sys
import json
from operator import add
from functools import reduce


class Neuron:
    def __init__(self, weights):
        self.ins = len(weights)
        self.weights = weights

    def compute(self, x):
        return reduce(add, map(lambda xi, wi: xi * wi, zip(x, self.weights)))


class Layer:
    def __init__(self, mat):
        self.ins = len(mat)
        self.outs = len(mat[0])
        self.neurons = [Neuron(list(map(lambda idx: mat[idx][neur], range(len(mat))))) for neur in range(len(mat[0]))]


class Network:
    def __init__(self, mats):
        self.layers = [Layer(mat) for mat in mats]
        self.ins = len(mats[0])
        self.outs = len(mats[-1][0])

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

    # def from_json(self, obj):
    #     self.ins = obj['ins']
    #     self.outs = obj['outs']
    #     self.layers = []


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


def main():
    if sys.argv[1] == '-i':
        net = Network(parse_txt(sys.argv[2]))
        with open('nn.json', 'w') as f:
            json.dump(net.to_json(), f, indent=4)


if __name__ == '__main__':
    main()
