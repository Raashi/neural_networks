import random
from math import exp
from utils import *

DEFAULT_TRAIN_ITERATIONS = 1

ACTIVATION_ALPHA = 2
TETTA = 0.1


def func_activate(x):
    return 2 / (1 + exp(-ACTIVATION_ALPHA * x)) - 1


def derivative_func_activate(x):
    return 2 * ACTIVATION_ALPHA * exp(-ACTIVATION_ALPHA * x) / ((1 + exp(-ACTIVATION_ALPHA * x)) ** 2)


class Neuron:
    def __init__(self, weights):
        self.ins = len(weights)
        self.weights = weights

    def compute(self, x):
        y = func_activate(reduce(add, map(lambda xw: xw[0] * xw[1], zip(x, self.weights))))
        Printer.neuron_computation(x, self.weights, y)
        return y

    def train(self, x):
        s = reduce(add, map(lambda xw: xw[0] * xw[1], zip(x, self.weights)))
        y = func_activate(s)
        return s, y


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

    def train(self, x):
        return [neuron.train(x) for neuron in self.neurons]


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

    def to_mats(self):
        mats = []
        for layer in self.layers:
            mat = [[0] * layer.outs for i in range(layer.ins)]
            for i in range(layer.ins):
                for j in range(layer.outs):
                    mat[i][j] = layer.neurons[j].weights[i]
            mats.append(mat)
        return mats

    def from_mats(self, mats):
        for idx, mat in enumerate(mats):
            for i in range(len(mat)):
                for j in range(len(mat[0])):
                    self.layers[idx].neurons[j].weights[i] = mat[i][j]

    def train(self, xd, train_iterations):
        if len(xd[0][0]) != self.ins:
            raise ValueError('Неверная размерность входного вектора обучающей выборки. '
                             'Должна быть {}'.format(self.ins))
        if len(xd[0][1]) != self.outs:
            raise ValueError('Неверная размерность выходного вектора обучающей выборки. '
                             'Должна быть {}'.format(self.outs))

        w_count = reduce(add, map(lambda layer: layer.outs, self.layers))
        if not (2 * w_count <= len(xd) <= 5 * w_count):
            Printer.train_25_rule(len(xd), w_count)

        for train_num in range(train_iterations):
            random.shuffle(xd)
            print('Старт итерации {}'.format(train_num + 1))

            w = self.to_mats()
            for kek, (xi, di) in enumerate(xd):
                # print('Тренировка на паре {}'.format(kek + 1))
                s = []
                y = [xi]
                for layer_num, wk in enumerate(w):
                    s.append([]), y.append([])
                    for j in range(len(wk[0])):
                        s[-1].append(reduce(add, map(lambda i: y[-2][i] * wk[i][j], range(len(wk)))))
                        y[-1].append(func_activate(s[-1][-1]))

                dw = 0.5 * reduce(add, map(lambda yd: (yd[0] - yd[1]) ** 2, zip(y[-1], di)))
                if PRINT_DEBUG:
                    print('Ошибка равна {:.3f}'.format(dw))

                y = y[1:]
                deltas = []
                for layer_num in range(len(w) - 1, -1, -1):
                    wk = w[layer_num]
                    yk = y[layer_num]
                    sk = s[layer_num]
                    ins = len(wk)
                    outs = len(wk[0])

                    if layer_num == len(w) - 1:
                        deltas.insert(0, [(yi - di) * derivative_func_activate(si) for di, si, yi in zip(di, sk, yk)])
                    else:
                        deltas_new = []
                        for j in range(outs):
                            delta = 0
                            outs_next = len(deltas[0])
                            for i in range(outs_next):
                                delta += deltas[0][i] * w[layer_num + 1][j][i]
                            delta *= derivative_func_activate(sk[j])
                            # print("{:.3f}".format(delta))
                            deltas_new.append(delta)
                        deltas.insert(0, deltas_new)

                    y_last = normalized(xi) if layer_num == 0 else y[layer_num - 1]
                    for i in range(ins):
                        for j in range(outs):
                            delt = -TETTA * deltas[0][j] * y_last[i]
                            if PRINT_DEBUG_FULL:
                                print('{:.3f}'.format(delt))
                            wk[i][j] += delt
                            wk[i][j] = max(min(100, wk[i][j]), -100)
            self.from_mats(w)

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
                    'weights': list(map(float_str, neuron.weights))
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
                    mat[idx][neuron_idx] = float(w)
            self.layers.append(Layer(mat))


def main():
    if sys.argv[1] == '-i':
        net = Network(parse_txt(sys.argv[2]))
        with open('nn.json', 'w') as f:
            json.dump(net.to_json(), f, indent=2)
    elif sys.argv[1] == '-c':
        net = Network(parse_json(sys.argv[2]))
        x = parse_x(sys.argv[3])
        print('Входной вектор x = ({})'.format(float_arr_str(x)))
        y = net.compute(x)
        print('Результат вычислений y = ({})'.format(float_arr_str(y)))
    elif sys.argv[1] == '-t':
        net = Network(parse_json(sys.argv[2]))
        xy = parse_xy(sys.argv[3])
        train_iterations = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_TRAIN_ITERATIONS

        net.train(xy, train_iterations)

        with open('nnt.json', 'w') as f:
            json.dump(net.to_json(), f, indent=2)
    else:
        print('Неверный код операции')


if __name__ == '__main__':
    main()
    # print("{:.3f}".format(derivative_func_activate(1)))
