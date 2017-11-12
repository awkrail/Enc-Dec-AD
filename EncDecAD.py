import chainer
from chainer import Variable, optimizers, serializers, Chain
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import numpy as np

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

    def learn(self):
        data_num = self.train_source.shape[0]
        for i in range(data_num):
            one_line = self.train_source[i]
            self.H.reset_state()
            self.zerograds()
            # calculate train loss
            loss = self.loss(one_line)
            loss.backward()
            loss.unchain_backward()
            self.optimizer.update()
            print("{0} / {1} line finishied".format(i+1, data_num))
            print("final loss", loss.data)

    def loss(self, one_line):
        # Encoder Side
        # calculate all h_t
        # last h_t is used for first decoder h_t initialization
        bar_h_i_list = self.encoder_h_i_list(one_line)
        # c_t = self.c_t(bar_h_i_list, last_h_i)

        # Decoder Side
        # first_decode is made with last_h_i and W
        length = one_line.shape[0]
        last_h_i = bar_h_i_list[-1]
        bar_x_i_list = self.decoder_x_i_list(last_h_i,length, test=False)

        # calculate the loss
        # loss is defined mean squared loss input and decoder x_i
        accum_loss = None
        for x_i, dec_x_i in zip(one_line, bar_x_i_list):
            x_i = np.array([[x_i]], dtype=np.float32)
            loss = F.mean_squared_error(x_i, dec_x_i)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss

    def encoder_h_i_list(self, line, test=False):
        h_i_list = []
        volatile = 'on' if test else 'off'
        for data in line:
            # dataはバッチ処理に対応させるために二次元にする必要がある
            h_i = self.H(Variable(np.array([[data]], dtype=np.float32), volatile=volatile))
            h_i_list.append(np.copy(h_i.data[0]))
        return h_i_list

    def decoder_x_i_list(self,last_h_i, length, test=False):
        decode_x_i_list = []
        x_i = self.W(Variable(np.array([last_h_i], dtype=np.float32), volatile=test))
        decode_x_i_list.append(x_i)
        for i in range(1, length):
            h_i = self.H(x_i)
            x_i = self.W(h_i)
            decode_x_i_list.append(x_i)
        return reversed(decode_x_i_list)

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
