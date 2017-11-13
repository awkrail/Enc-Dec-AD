from EncDecAD import EncDecAD
import numpy as np
import argparse

import chainer
from chainer import cuda
from chainer import serializers

# use gpu or not
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID')
parser.add_argument('--epoch', '-e', default=50, type=int,
                    help='train epoch number')
parser.add_argument('--batch_size', '-b', default=20, type=int,
                    help='batch processing number')
args = parser.parse_args()

# model
model = EncDecAD()

# use cuuda
xp = cuda.cupy if args.gpu >= 0 else np
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# train
epoch_num = args.epoch
for epoch in range(epoch_num):
    print('{0} / {1} epoch start'.format(epoch+1, epoch_num))
    model.learn(args.batch_size)
    modelfile = 'model/encdec' + str(epoch) + '.model'
    serializers.save_npz(modelfile, model)
    print('{0} / {1} epoch finished'.format(epoch+1, epoch_num))
