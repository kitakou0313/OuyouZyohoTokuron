import numpy as np

def logistic(x):
    y = 1 / (1 + np.exp(-x))
    return y

ETA = 1.0
#学習率
ALPHA = 1.0
#安定化係数
MIDDLE_LAYER_WIDTH = 50
#中間層の数

class NeuralNet(object):
    """
    NeuralNetworkクラス
    入力:64次元ベクトル（メッシュ特徴の次元数）+ 1次元(1で固定)
    中間層:コンストラクタの引数で指定
    出力:20次元ベクトル（文字の種類と同数）
    """
    def __init__(self) -> None:
        self.Wji:np.ndarray = np.array([])
        self.Wkj:np.ndarray = np.array([])

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
