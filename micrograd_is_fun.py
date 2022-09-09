import math
import random

# TODO: add tensors WITH FUCKING * BY SCALARS
# TODO: train and test on MINST or similar
# TODO: implement convolutional


class Value:

    def __init__(self, data, _connected=(), _operator="", label=""):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._preview = set(_connected)
        self._operator = _operator
        self.label = label

    def __repr__(self):
        return f"czarna magia kurwa(data = {self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * output.grad
            other.grad += 1.0 * output.grad
        output._backward = _backward
        return output

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
        output._backward = _backward
        return output

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        output = Value(self.data**other, (self, ), f"^{other}")

        def _backward():
            self.grad += other * (self.data ** (other-1)) * output.grad
        output._backward = _backward
        return output

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def exp(self):
        x = self.data
        output = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += output.data * output.grad
        output._backward = _backward
        return output


    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        output = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * output.grad

        output._backward = _backward
        return output

    def relu(self):
        output = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (output.data > 0) * output.grad
        output._backward = _backward

        return output

    def backward(self):
        topological_order = []
        visited = set()

        def loop(v):
            if v not in visited:
                visited.add(v)
                for child in v._preview:
                    loop(child)
                topological_order.append(v)
        loop(self)
        self.grad = 1
        for node in reversed(topological_order):
            node._backward()


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class model(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


xs = [
    [1.0, 5.0, -1.0]
]

ys = [1.0]


n = model(3, [4,4,1])

for k in range(20):

    # forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    # backward pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    # update
    for p in n.parameters():
        p.data += -0.1 * p.grad

    print(k, loss.data)
