import numpy as np

def logistic(x):
    y = 1 / (1 + np.exp(-x))
    return y

class NeuralNet(object):
    """
    NeuralNetworkクラス
    入力:64次元ベクトル（メッシュ特徴の次元数）+ 1次元(1で固定)
    中間層:コンストラクタの引数で指定
    出力:20次元ベクトル（文字の種類と同数）
    """
    def __init__(self, numOfMiddleLayers:int) -> None:
        self.layer = []

    def train(self) -> None:
        """
        学習用関数
        """
        pass
    def forward(self):
        """
        識別用関数
        """
        pass
    def backForward(self):
        """
        重みの更新関数
        """
        pass
