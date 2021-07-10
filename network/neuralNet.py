import numpy as np
from utils import dataparser

def logistic(x):
    y = 1 / (1 + np.exp(-x))
    return y

ETA = 0.01
#学習率

ALPHA = 1.0
#安定化係数

INPUT_LAYER_WIDTH = 64
MIDDLE_LAYER_WIDTH = 50
OUTPUT_LAYER_WIDTH = 20 

#中間層の数(1の出力は含まない)

class NeuralNet(object):
    """
    NeuralNetworkクラス
    入力:64次元ベクトル（メッシュ特徴の次元数）+ 1次元(1で固定)
    中間層:コンストラクタの引数で指定
    出力:20次元ベクトル（文字の種類と同数）
    """
    def __init__(self) -> None:
        #定数1を追加する用の次元を用意する
        inputLayerWidthWith1 = INPUT_LAYER_WIDTH + 1
        middleLayerWidth1With1 = MIDDLE_LAYER_WIDTH + 1

        #標準偏差（平均0、標準偏差1）で初期化
        self.Wji:np.ndarray = np.random.randn(MIDDLE_LAYER_WIDTH,inputLayerWidthWith1)
        self.Wkj:np.ndarray = np.random.randn(OUTPUT_LAYER_WIDTH,middleLayerWidth1With1)

    def train(self,charData:dataparser.CharData) -> None:
        """
        学習用関数
        """
        resVec = self.forward(charData)
        pass

    def forward(self, charData:dataparser.CharData) -> np.ndarray:
        """
        識別用関数
        """
        inputVector = charData.meshFeature

        #中間層
        inputVectorWith1 = np.insert(inputVector, len(inputVector), 1)
        u = self.Wji @ inputVectorWith1
        yj = logistic(u)

        #出力層
        yj1 = np.insert(yj, len(yj), 1)
        u = self.Wkj @ yj1
        yk = logistic(u)

        return yk
        
    def backForward(self):
        """
        重みの更新関数
        """
        pass
