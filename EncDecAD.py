import chainer
from chainer import Variable, optimizers, serializers, Chain
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import serializers

import numpy as np
import argparse

class EncDecAD(chainer.Chain):
    def __init__(self, train_source="data/anormaly_data/train_and_test/train.npy", test_source="data/anormaly_data/train_and_test/test.npy"):
        self.train_source = np.load(train_source)
        self.test_source = np.load(test_source)
        super(EncDecAD, self).__init__(
                H = L.LSTM(1, 30),
                Wc1 = L.Linear(30, 30),
                Wc2 = L.Linear(30, 30),
                W = L.Linear(30, 1)
                )
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self)

    def learn(self, batch_size):
        data_num = self.train_source.shape[0]
        sffindx = np.random.permutation(data_num)
        for j in range(0, data_num, batch_size):
            x = self.train_source[sffindx[j:(j+batch_size) if (j+batch_size) < data_num else data_num]]
            self.H.reset_state()
            self.zerograds()
            # calculate batch train loss
            loss = self.loss(x)
            loss.backward()
            loss.unchain_backward()
            self.optimizer.update()

            """
            one_line = self.train_source[i]
            self.H.reset_state()
            self.zerograds()
            # calculate train loss
            loss = self.loss(one_line)
            loss.backward()
            loss.unchain_backward()
            self.optimizer.update()
            print("{0} / {1} line finishied".format(i+1, data_num))
            """
            print("final loss", loss.data)

    def calc_gaussian_params(self):
        for one_line in self.test_source:
            for x in one_line:
                # encoderに通してdecoderされてきたもの
                pass

    def loss(self, x):
        # Encoder Side
        # calculate all h_t
        # last h_t is used for first decoder h_t initialization
        bar_h_i_list = self.encoder_h_i_list(x)
        # c_t = self.c_t(bar_h_i_list, last_h_i)

        # Decoder Side
        # first_decode is made with last_h_i and W
        # if you don't use batch_size, you can use x.shape[0]
        length = x.shape[1]
        last_h_i = bar_h_i_list[-1]
        bar_x_i_list = self.decoder_x_i_list(last_h_i, length, test=False)

        # calculate the loss
        # loss is defined mean squared loss input and decoder x_i
        accum_loss = None
        row = x.shape[0]
        col = x.shape[1]
        for i in range(col):
            x_i = x[:, i].reshape(row, 1).astype(np.float32)
            dec_x_i = bar_x_i_list[i].data.astype(np.float32)
            loss = F.mean_squared_error(x_i, dec_x_i)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        """ 
        for x_i, dec_x_i in zip(x, bar_x_i_list):
            import ipdb; ipdb.set_trace()
            x_i = np.array([[x_i]], dtype=np.float32)
            loss = F.mean_squared_error(x_i, dec_x_i)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        """
        return accum_loss

    def encoder_h_i_list(self, line, test=False):
        h_i_list = []
        volatile = 'on' if test else 'off'
        row = line.shape[0]
        col = line.shape[1]
        for i in range(col):
            h_i = self.H(Variable(xp.array(line[:, i].reshape(row, 1), dtype=np.float32)))
            h_i_list.append(xp.copy(h_i.data))
            # これで, 各データが時系列で順番に入った時の, LSTMの内部状態が入る
        """
        for data in line:
            # dataはバッチ処理に対応させるために二次元にする必要がある
            h_i = self.H(Variable(np.array([[data]], dtype=np.float32), volatile=volatile))
            h_i_list.append(np.copy(h_i.data[0]))
        """
        return h_i_list

    def decoder_x_i_list(self,last_h_i, length, test=False):
        decode_x_i_list = []
        #x_i = self.W(Variable(np.array([last_h_i], dtype=np.float32), volatile=test))
        x_i = self.W(last_h_i)
        decode_x_i_list.append(x_i)
        for i in range(1, length):
            h_i = self.H(x_i)
            x_i = self.W(h_i)
            decode_x_i_list.append(x_i)
        return list(reversed(decode_x_i_list))

    def c_t(self, bar_h_i_list, h_t, test=False):
        s = 0.0
        #import ipdb; ipdb.set_trace()
        for bar_h_i in bar_h_i_list:
            s += np.exp(h_t.dot(bar_h_i))
        c_t = np.zeros(30)

        for bar_h_i in bar_h_i_list:
            alpha_t_i = np.exp(h_t.dot(bar_h_i)) / s
            c_t += alpha_t_i * bar_h_i

        volatile = 'on' if test else 'off'
        c_t = Variable(np.array([c_t]).astype(np.float32), volatile=volatile)
        return c_t


# use gpu or not
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID')
parser.add_argument('--epoch', '-e', default=50, type=int, help='train epoch number')
parser.add_argument('--batch_size', '-b', default=20, type=int, help='batch processing number')
args = parser.parse_args()

# model
model = EncDecAD()

# use cuda
xp = cuda.cupy if args.gpu >= 0 else np
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# train
epoch_num = args.epoch
for epoch in range(epoch_num):
    print('{0} / {1} epoch start'.format(epoch+1, epoch_num))
    model.learn(args.batch_size)
    modelfile = 'model/encdec-' + str(epoch) + '.model'
    serializers.save_npz(modelfile, model)
    print('{0} / {1} epoch finished'.format(epoch+1, epoch_num))


if __name__ == "__main__":
    model = EncDecAD()
    epoch_num = 100

    for epoch in range(epoch_num):
        print("{0} / {1} epoch start".format(epoch+1, epoch_num))

        # start training
        model.learn()
        modelfile = "encdec" + str(epoch) + ".model"
        model.save_model(modelfile)
        
        print("{0} / {1} epoch finishied".format(epoch+1, epoch_num))
