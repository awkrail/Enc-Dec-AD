import chainer
from chainer import Variable, optimizers, serializers, Chain
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import serializers

import numpy as np
import argparse

class EncDecAD(chainer.Chain):
    def __init__(self, train_source="data/anormaly_data/train_and_test/train.npy", test_source="data/anormaly_data/train_and_test/test.npy", hidden_n=30, gpu=-1):
        self.gpu=gpu
        xp = cuda.cupy if self.gpu >= 0 else np
        self.train_source = xp.load(train_source)
        self.test_source = xp.load(test_source)
        super(EncDecAD, self).__init__(
                H = L.LSTM(1, hidden_n),
                Wc1 = L.Linear(hidden_n, hidden_n),
                Wc2 = L.Linear(hidden_n, hidden_n),
                W = L.Linear(hidden_n, 1)
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
            loss = self.loss(x)
            loss.backward()
            self.optimizer.update()
            print("final loss", loss.data)

    def load_model(self, path):
        serializers.load_npz(path, self)

    def calc_gaussian_params(self):
        e_i_list = []
        # debugの時は全部で計算しない方がよさそう..
        for i, one_line in enumerate(self.test_source):
            print('now {} line processing..'.format(i))
            batch_one_line = one_line.reshape(1, one_line.shape[0])
            h_i_list = self.encoder_h_i_list(batch_one_line) 
            length = batch_one_line.shape[1]
            last_h_i = h_i_list[-1]
            x_i_list = [x_i.data[0][0] for x_i in self.decoder_x_i_list(last_h_i, length, test=True)]
            x_i_list = np.array(x_i_list, dtype=np.float32)
            # calc |x_i - x_i(dec)|
            abs_e_i = np.abs((one_line - x_i_list))
            e_i_list.append(abs_e_i)

        # e_i => μ, Σ
        # 縦に計算する
        e_i_np = np.array(e_i_list, dtype=np.float32)
        
        # 要確認
        mu = np.mean(e_i_np, axis=0)
        mu_T = np.array([mu], dtype=np.float32).T
        cov = np.zeros((mu.shape[0], mu.shape[0]))
        for e_i in e_i_np:
            e_i_T = np.array([e_i], dtype=np.float32).T
            sig = np.dot((e_i_T-mu_T), (e_i_T-mu_T).T)
            cov += sig
        sigma = cov / e_i_np.shape[0]
        return mu, sigma

    # calculate anormaly score (X-μ)^TΣ^(-1)(X-μ)
    def score(self, valid_X, mu, sigma):
        mu_T = np.array([[mu]], dtype=np.float32).T
        inv_sigma = np.linalg.inv(sigma) # sigmaに逆行列がない..なんてことにはならない?
        for one_line in valid_X:
            batch_one_line = one_line.reshape(1, one_line.shape[0])
            h_i_list = self.encoder_h_i_list(batch_one_line)
            length = batch_one_line.shape[1]
            last_h_i = h_i_list[-1]
            x_i_list = [x_i.data[0][0] for x_i in self.decoder_x_i_list(last_h_i, length, test=True)]
            x_i_list_T = np.array(x_i_list, dtype=np.float32).T
            e_minus_mu = x_i_list_T - mu_T
            yield np.dot((e_minus_mu.T, inv_sigma), e_minus_mu)

    def loss(self, x):
        xp = cuda.cupy if self.gpu >= 0 else np
        # Encoder Side
        # calculate all h_t
        # last h_t is used for first decoder h_t initialization
        bar_h_i_list = self.encoder_h_i_list(x)
        last_h_i = bar_h_i_list[-1]
        # attention
        c_t = self.c_t(bar_h_i_list[:-1], last_h_i)
        bar_h_t = F.tanh(self.Wc1(c_t) + self.Wc2(last_h_i))

        # Decoder Side
        # first_decode is made with bar_h_i and W
        bar_x_i_list = self.decoder_x_i_list(bar_h_t, bar_h_i_list)
        
        # calculate the loss
        # loss is defined mean squared loss input and decoder x_i
        accum_loss = None
        row = x.shape[0]
        col = x.shape[1]
        for i in range(col):
            x_i = x[:, i].reshape(row, 1).astype(xp.float32)
            dec_x_i = bar_x_i_list[i].data.astype(xp.float32)
            loss = F.mean_squared_error(x_i, dec_x_i)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss

    def encoder_h_i_list(self, line):
        xp = cuda.cupy if self.gpu >= 0 else np
        h_i_list = []
        row = line.shape[0]
        col = line.shape[1]
        for i in range(col):
            h_i = self.H(Variable(xp.array(line[:, i].reshape(row, 1), dtype=xp.float32)))
            h_i_list.append(xp.copy(h_i.data))
        return h_i_list

    def decoder_x_i_list(self, bar_h_t, bar_h_i_list):
        decode_x_i_list = []
        x_i = self.W(bar_h_t)
        length = len(bar_h_i_list)
        decode_x_i_list.append(x_i)
        for i in range(1, length):
            h_i = self.H(x_i)
            c_t = self.c_t(bar_h_i_list, h_t.data)
            bar_h_t = F.tanh(self.Wc1(c_t)+self.Wc2(h_i))
            x_i = self.W(bar_h_t)
            decode_x_i_list.append(x_i)
        return list(reversed(decode_x_i_list))

    def c_t(self, bar_h_i_list, h_t):
        pass
