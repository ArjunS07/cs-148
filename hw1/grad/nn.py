"""
tiny Neural Network library built on top of your autograd implementation.
"""

from grad import Value
import random  # use random.uniform to generate random weights and biases


class Module(object):
    def nparams(self) -> int:
        return len(self.params())

    def params(self) -> list[Value]:
        return []

    def step(self, lr) -> None:
        # iterate through every parameter, and then update it
        for p in self.params():
            p = p - lr * p.grad

    def zero_grad(self) -> None:
        for p in self.params():
            p.zero_grad()

    def __repr__(self) -> str:
        return super().__repr__()


class Perceptron(Module):
    """
    TODO: implement a perceptron that takes in an input vector in R^n and outputs a single value
    Perceptron: R^n -> R
    (1) x is a list with length N (representing a vector in R^n)
    (2) signal = x_1*w_1 + x_2*w_2 + ... + x_N*w_N
    (3) activate = activation_fn (s) where s = signal(x)
    NOTE: activation_fn can be "relu" "sigmoid" or "tanh"
    """

    def __init__(self, N: int, activation_fn="relu"):
        self.N = N  # num of inputs to this perceptron
        self.w: list[Value] = (
            None  # TODO: initialize weights using random.uniform(-1, 1) (length N)
        )
        for _ in range(N):
            self.w.append(Value(random.uniform(-1, 1)))
        self.b: Value = Value(0)  # TODO: initialize bias as 0 (single number)
        self.activation_fn = activation_fn

    def signal(self, x: list[Value]) -> Value:
        s = self.b
        for i in range(self.N):
            s = s + x[i] * self.w[i]
        return s

    def activate(self, s: Value) -> Value:
        if self.activation_fn == "relu":
            f_relu = lambda x: x if x.data > 0 else Value(0)
            return f_relu(s)
        elif self.activation_fn == "sigmoid":
            f_sigmoid = lambda x: 1 / (1 + (-x).exp())
            return f_sigmoid(s)
        elif self.activation_fn == "tanh":
            f_tanh = lambda x: (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())
            return f_tanh(s)
        else:
            raise Exception(f"Unknown activation function: {self.activation_fn}")

    def params(self) -> list[Value]:
        # TODO: return a list of all parameters of the Neuron
        # hint: weights and biases
        params = []
        for weight in self.w:
            params.append(weight)
        params.append(self.b)
        return params

    def __call__(self, x) -> Value:
        """
        Example usage:
        >>> p = Perceptron(3) # notice N=3
        >>> p([1, 2, 3]) # notice input is a vector in R^3
        """
        if len(x) != self.N:
            raise Exception(f"Expected vector of length: {self.N}")
        return self.activate(self.signal(x))

    def __repr__(self) -> str:
        return f"Neuron({self.N})"


class Layer(Module):
    """
    TODO: implement a perceptron layer
    Layer: R^N_in -> R^N_out
    (1) input x is a list with length N_in (representing a vector in R^N_in)
    (2) a layer has R^N_out perceptrons
    (3) each perceptron spits out a single number in R
    (4) put them all together in an output vector/list of length N_out
    """

    def __init__(self, N_in, N_out, activation_fn: str) -> None:
        self.N_in = N_in  # number of input features
        self.N_out = N_out  # number of output features
        self.perceptrons = [Perceptron(N_in, activation_fn) for _ in range(N_out)]

    def __call__(self, x) -> Value | list[Value]:
        assert len(x) == self.N_in
        out: list[Value] = []
        for p in self.perceptrons:
            out.append(p(x))
        if len(out) == 1:
            # NOTE: if out is a single number just return it. Ex: [n] -> n
            return out[0]
        else:
            return out

    def params(self) -> list[Value]:
        # TODO: return a list of all params in this Layer, hint: you can reuse Perceptron's params method
        params = []
        for p in self.perceptrons:
            params.extend(p.params())
        return params

    def __repr__(self) -> str:
        return f"Layer(N_in={self.N_in}, N_out={self.N_out})"


class MLP(Module):
    """
    Implement an MLP (multi layer perceptron)
    (1) You are given N_in so x will be a vector in R^N_in
    (2) N_outs is a list of integers (the number of outputs of each layer)
    (3) Example:
        - if N_in=10 and N_outs=[12, 16, 14, 3] then self.N = [10] + [12, 16, 14, 3] = [10, 12, 16, 14, 3]
        - input in R^10
        - layer #1: R^10 -> R^12
        - layer #2: R^12 -> R^16
        - layer #3: R^16 -> R^14
        - layer #4: R^14 -> R^3
    (4) Your job is to initialize these layers in self.layers

    Example Usage
    -------------
    >>> model = MLP(3, [10, 10, 2], "relu")
    >>> model([1, 2, 3])
    """

    def __init__(self, N_in: int, N_outs: list[int], activation_fn: str):
        self.N = [N_in] + N_outs
        # TODO: initialize a list of Layer's, with the correct in/out dims for each one (hint: self.N has been assigned for you)
        layers = []
        for _ in range(len(self.N) - 1):
            layers.append(Layer(self.N[_], self.N[_ + 1], activation_fn))
        self.layers = layers

    def __call__(self, x) -> Value | list[Value]:
        for layer in self.layers:
            x = layer(x)
        return x

    def params(self) -> list[Value]:
        # TODO: return a list of all params in this MLP, hint: you can reuse Layer's params method
        params = []
        for layer in self.layers:
            params.extend(layer.params())
        return params
    
    def __repr__(self) -> str:
        # prints out mlp as a readable string
        return f"MLP({self.N}): {[str(l) for l in self.layers]}]"
