import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

from IPython import embed

f = L.Linear(3, 2)
x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
y = f(x)
# y.data

f.cleargrads()
y.grad = np.ones((2, 2), dtype=np.float32)
y.backward()
# f.W.grad
# f.b.grad
