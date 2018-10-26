import sys
import json
import math
import operator
import functools


FILENAME_INPUT_DEFAULT = '.\\tests\\graph.txt'
FILENAME_MARKS_DEFAULT = '.\\tests\\graph_marks.txt'
FILENAME_OUTPUT_DEFAULT = 'graph_serialized.json'


class GraphInputError(SyntaxError):
    pass


class Graph:
    def __init__(self, fp):
        self.edges_count = 0
        self.vertices = set()
        self.backward = dict()
        self.forward = {}

        self._read(fp.readline().replace(' ', ''))
        self.sort()

    def add_edge(self, a, b, n):
        if b not in self.backward:
            self.backward[b] = []
        if n in map(lambda t: t[0], self.backward[b]):
            raise GraphInputError('Повторяющийся номер входящей дуги: {}, {}, {}'.format(a, b, n))
        if any(map(lambda v: v[1] == a, self.backward[b])):
            raise GraphInputError('Такая дуга уже есть в графе: ({} - {})'.format(a, b))
        self.backward[b].append((n, a))

        if a not in self.forward:
            self.forward[a] = []
        self.forward[a].append(b)

        self.edges_count += 1
        self.vertices.add(a)
        self.vertices.add(b)

    def sort(self):
        for b, edge in self.backward.items():
            edge.sort(key=lambda e: e[0])

    def serialize(self):
        self.sort()
        return self.backward

    def _read(self, line):
        last_rbracket = -1
        idx_bracket = 0
        while line.find('(', last_rbracket + 1) != -1:
            try:
                lbracket = line.find('(', last_rbracket + 1)
                rbracket = line.find(')', last_rbracket + 1)
                if rbracket < 0:
                    raise GraphInputError('Не найдена закрывающая скобка. Последняя открывающая {}'.find(lbracket))
                idx_bracket += 1
                last_rbracket = rbracket
                line_edge = line[lbracket + 1: rbracket]
                self._read_edge(line_edge)
            except GraphInputError as e:
                raise GraphInputError('Ошибка в скобке {}'.format(idx_bracket)) from e
        if last_rbracket + 1 < len(line):
            raise GraphInputError('Ошибка при считывании - конец строки')

    def _read_edge(self, line_edge):
        values = line_edge.split(',')
        if len(values) < 3:
            raise GraphInputError('Неверный формат строки входного файла')

        a, b = values[0], values[1]
        try:
            n = int(values[2])
        except ValueError:
            raise GraphInputError('Неверный формат номера дуги: должно быть целым числом')
        self.add_edge(a, b, n)


class GraphExInputError(ValueError):
    pass


class GraphEx(Graph):
    def get_sources(self):
        return list({a for b, blist in self.backward.items() for (n, a) in blist if a not in self.backward})

    def get_sinks(self):
        return list({a for a in self.vertices if a not in self.forward})

    def assert_cycle(self):
        def rec(stack):
            a = stack[-1]
            if a not in self.forward:
                return
            for b in self.forward[a]:
                if b in stack:
                    raise GraphInputError('Обнаружен цикл: {}'.format(stack[stack.index(b):]))
                if b not in visited:
                    visited.add(b)
                    rec(stack + [b])

        sources = self.get_sources()
        visited = set()
        for source in sources:
            rec([source])

    def function(self) -> str:
        def rec(b):
            rec_value = b + '('
            if b not in self.backward:
                return b
            for (n, a) in self.backward[b]:
                rec_value += rec(a) + ','
            return rec_value[:-1] + ')'

        self.assert_cycle()
        sinks = self.get_sinks()
        if len(sinks) > 1:
            raise GraphInputError('Функция от графа определена только для графов с одним стоком')
        return rec(sinks[0])

    @staticmethod
    def read_marks(fp):
        marks = {}
        for line in fp:
            vertex, op = line.strip().replace(' ', '').split(':')
            marks[vertex] = op
        return marks

    def compute(self, marks: dict):
        if set(marks.keys()) != self.vertices:
            raise GraphExInputError('Операторы/значения переданы не для всех вершин')

        value = self.function()
        for mark, op in marks.items():
            value = value.replace(mark, op)

        def compute_expr(expr_op, expr_values):
            if expr_op == '+':
                if not len(expr_values):
                    raise GraphExInputError('Оператор + принимает ненулевое число аргументов')
                return functools.reduce(operator.add, expr_values)
            elif expr_op == '*':
                if not len(expr_values):
                    raise GraphExInputError('Оператор * принимает ненулевое число аргументов')
                return functools.reduce(operator.mul, expr_values)
            elif expr_op == 'exp':
                if len(expr_values) != 1:
                    raise GraphExInputError('Оператор exp принимает только один аргумент')
                return math.exp(expr_values[0])

        def compute_rec(substr: str):
            if '(' not in substr:
                return int(substr)
            rec_op = substr[:substr.find('(')]
            rec_values = substr[substr.find('(') + 1: -1].split(',')
            rec_values = [compute_rec(v) for v in rec_values]
            return compute_expr(rec_op, rec_values)

        return compute_rec(value)


def main():
    operation = sys.argv[1]
    file_graph = FILENAME_INPUT_DEFAULT if len(sys.argv) < 3 else sys.argv[2]

    if operation == '-parse':
        with open(file_graph) as f:
            graph = Graph(f)
        with open(FILENAME_OUTPUT_DEFAULT, 'w') as f:
            json.dump(graph.serialize(), f, indent=4, sort_keys=True)
    elif operation == '-func':
        with open(file_graph) as f:
            graph = GraphEx(f)
        print(graph.function())
    elif operation == '-comp':
        with open(file_graph) as f:
            graph = GraphEx(f)
        file_marks = FILENAME_MARKS_DEFAULT if len(sys.argv) < 4 else sys.argv[3]
        with open(file_marks) as f:
            marks = GraphEx.read_marks(f)
        print(graph.compute(marks))
    else:
        print('Неверный параметр операции')


if __name__ == '__main__':
    main()
