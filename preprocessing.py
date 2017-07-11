import csv
import random
import numpy as np

"""
    学習データ, テストデータ, バリデーションデータに分けるクラス
    EncDecADでは, 異常では無いnormalデータを学習させて, 波形を出力する。
    訓練データはEncoderDecoderを学習させるためのクラス。72%
    テストデータは平均や分散を計算するためのクラス。 9%
    バリデーションデータは実際にデータを入れるとき用のクラスである。(ここにのみ異常値が入ってくる)
"""


class PreprocessClass(object):
    def __init__(self):
        self.csv0 = []
        self.train_csv0 = []
        self.test_csv0 = []
        self.validation_csv0 = []
        self.anomaly = []

    def load_csv(self):
        with open('ex_log/Raw/0.csv', 'r') as f:
            self.csv0 = [np.array(row[:50], dtype=np.float32) for row in csv.reader(f)]

    def divide_train_and_test(self):
        random.shuffle(self.csv0)
        threshold = int(len(self.csv0)/10*9)
        self.train_csv0 = self.csv0[0:threshold]
        self.validation_csv0 = self.csv0[threshold:]

        # thresholdを再設定して, 学習データをさらに分ける
        threshold = int(len(self.train_csv0)/5*4)
        self.test_csv0 = self.train_csv0[threshold:]
        self.train_csv0 = self.train_csv0[0:threshold]

    def get_anomaly_data(self):
        with open('ex_log/Raw/1.csv', 'r') as f:
            self.anomaly = [np.array(row[:50], dtype=np.float32) for row in csv.reader(f)]






