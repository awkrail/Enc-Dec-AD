import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from preprocessing import PreprocessClass


class EncDecAdClass(chainer.Chain):
    def __init__(self, m, c, epoch):
        super().__init__(
            H=L.LSTM(m, c),
            W=L.Linear(c, m)
        )
        self.preprocess = PreprocessClass()
        self.preprocess.load_csv()
        self.preprocess.divide_train_and_test()
        self.train = self.preprocess.train_csv0
        self.epoch = epoch

    def forward(self):
        self.H.reset_state()
        h = None
        accum_loss = None
        for data50dim in self.train:
            for data in data50dim:
                h = self.H(np.array([[data]], dtype=np.float32))
            next_data = self.W(h)
            # import ipdb; ipdb.set_trace()
            accum_loss = F.mean_squared_error(next_data, np.array([[data50dim[-1]]], dtype=np.float32))  # 最初の二乗誤差
            for i in reversed(range(49)): # 一番最後のものは上で計算済み
                h = self.H(next_data)
                next_data = self.W(h)
                accum_loss += F.mean_squared_error(next_data, np.array([[data50dim[i]]], dtype=np.float32))

        return accum_loss


if __name__ == '__main__':
    # setup model
    epoch = 100
    model = EncDecAdClass(1, 10, epoch)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    for i in range(model.epoch):
        loss = model.forward()
        loss.backward()
        optimizer.update()
        print("%d epoch" % i, str(loss))











