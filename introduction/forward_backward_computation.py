import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

from IPython import embed

x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
y = x**2 - 2*x + 1
y.grad = np.ones((2, 3), dtype=np.float32)
y.backward()
x.grad
