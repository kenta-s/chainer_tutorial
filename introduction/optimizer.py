import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

from IPython import embed
from write_a_model_as_a_chain import MyChain

model = MyChain()
optimizer = optimizers.SGD()
optimizer.setup(model)

x = np.random.uniform(-1, 1, (2, 4)).astype('f')
model.cleargrads()
# compute gradient here...
loss = F.sum(model(chainer.Variable(x)))
loss.backward()
optimizer.update()

embed()
