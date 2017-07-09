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

# # Saving/Loading model
# serializers.save_npz('my.model', model)
# serializers.load_npz('my.model', model)

# # Saving/Loading optimizer
# serializers.save_npz('my.state', optimizer)
# serializers.load_npz('my.state', optimizer)
