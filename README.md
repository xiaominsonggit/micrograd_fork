
# micrograd

![flow chart forward back prop](docs/images/BackProp.png)

A tiny Autograd engine (with a bite! :)). Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are tiny, with about 100 and 50 lines of code respectively. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification, as the demo notebook shows. Potentially useful for educational purposes.

### Installation

```bash
pip install micrograd
```

### Example usage

Below is a slightly contrived example showing a number of possible supported operations:

```python
from micrograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
```

### Training a neural net

The notebook `demo.ipynb` provides a full demo of training an 2-layer neural network (MLP) binary classifier. This is achieved by initializing a neural net from `micrograd.nn` module, implementing a simple svm "max-margin" binary classification loss and using SGD for optimization. As shown in the notebook, using a 2-layer neural net with two 16-node hidden layers we achieve the following decision boundary on the moon dataset:

![2d neuron](moon_mlp.png)

### Tracing / visualization

For added convenience, the notebook `trace_graph.ipynb` produces graphviz visualizations. E.g. this one below is of a simple 2D neuron, arrived at by calling `draw_dot` on the code below, and it shows both the data (left number in each node) and the gradient (right number in each node).

```python
from micrograd import nn
n = nn.Neuron(2)
x = [Value(1.0), Value(-2.0)]
y = n(x)
dot = draw_dot(y)
```

![2d neuron](gout.svg)

### Running tests

To run the unit tests you will have to install [PyTorch](https://pytorch.org/), which the tests use as a reference for verifying the correctness of the calculated gradients. Then simply:

```bash
python -m pytest
```

### Chain Rule of Calculus
![chain_rule](docs/images/chain_rule.PNG)

This is the equation for the chain rule of calculus. Suppose z = f(g(x)) = f(y) , where z is typically the loss. The gradient of the loss ∂z/∂x is sum of the products of the local derivative  ∂y/∂x and the gradient of the loss with respect to y, ∂z/∂y.

In vector or matrix notation, the back-propagation algorithm consists of performing such a Jacobian-gradient product for each operation in the graph.

![chain_rule_vector](docs/images/chain_rule_vecotor.PNG)

The Jacobian represents the "local" partial derivatives, in contrast to the gradients with respect to all weights, which propagate backwards.

### Flow Chart of Forward and Backward Propagations

The first picture is a flow chart of forward and backward propagation for a single node. (Top) The data flows from left to right during forward propagation, and then the gradients flow from right to left during back-propagation. Notice the gradients accumulate from other operations in the same layer (depicted with dotted lines below).


### Common Pitfalls
The chain rule of calculus may appear clean and straightforward compared to traditional statistical models, but there are many pitfalls when implementing it.

[Common pitfalls](docs/pitfalls.md)

### License

MIT
