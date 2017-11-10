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

    def learn(self):
        data_num = self.train_source.shape[0]
        #import ipdb;
        #ipdb.set_trace()
        # 一つの行ごとに回す
        for i in range(data_num):
            one_line = self.train_source[i]
            self.H.reset_state()
            self.zerograds()
            # calculate train loss
            loss = self.loss(one_line)
            loss.backward()
            loss.unchain_backward()
            self.optimizer.update()

    def loss(self, one_line):
        # Encoder Side
        # calculate all h_t
        # last h_t is used for first decoder h_t initialization
        #import ipdb; ipdb.set_trace()
        bar_h_i_list = self.h_i_list(one_line)
        #import ipdb; ipdb.set_trace()
        last_h_i = bar_h_i_list[-1]
        c_t = self.c_t(bar_h_i_list, last_h_i)
        print(c_t)

    def h_i_list(self, line, test=False):
        h_i_list = []
        volatile = 'on' if test else 'off'
        #import ipdb; ipdb.set_trace()
        for data in line:
            # dataはバッチ処理に対応させるために二次元にする必要がある
            h_i = self.H(Variable(np.array([[data]], dtype=np.float32), volatile=volatile))
            h_i_list.append(np.copy(h_i.data[0]))
        return h_i_list

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
