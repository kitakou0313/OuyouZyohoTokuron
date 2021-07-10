from typing import List
import numpy as np
from numpy.random.mtrand import rand
from utils import dataparser

def logistic(x):
    y = 1 / (1 + np.exp(-x))
    return y

ETA = 0.2
#学習率

ALPHA = 0.9
#安定化係数

INPUT_LAYER_WIDTH = 64
MIDDLE_LAYER_WIDTH = 50
OUTPUT_LAYER_WIDTH = 20 

#中間層の数(1の出力は含まない)

class NeuralNet(object):
    """
    NeuralNetworkクラス
    入力:64次元ベクトル（メッシュ特徴の次元数）+ 1次元(1で固定)
    中間層:定数で指定
    出力:20次元ベクトル（文字の種類と同数）
    """
    def __init__(self) -> None:
        #定数1を追加する用の次元を用意する
        inputLayerWidthWith1 = INPUT_LAYER_WIDTH + 1
        middleLayerWidth1With1 = MIDDLE_LAYER_WIDTH + 1

        #標準偏差（平均0、標準偏差1）で初期化
        self.Wji:np.ndarray = np.random.randn(MIDDLE_LAYER_WIDTH,inputLayerWidthWith1)
        self.Wkj:np.ndarray = np.random.randn(OUTPUT_LAYER_WIDTH,middleLayerWidth1With1)

        #学習時の重み更新用の中間層の出力
        self.yj1:np.ndarray = np.array([])

        #学習安定用の前回学習時の更新幅
        self.dWji_t_1:np.ndarray = np.array([])
        self.dWkj_t_1:np.ndarray = np.array([])

    def train(self,charData:dataparser.CharData) -> None:
        """
        学習用関数
        """
        yk = self.forward(charData)
        yj1 = self.yj1
        yi = np.insert(charData.meshFeature, len(charData.meshFeature), 1)
        yk_hat = charData.ansLabel.ansVec

        K = OUTPUT_LAYER_WIDTH
        J = MIDDLE_LAYER_WIDTH + 1
        I = INPUT_LAYER_WIDTH + 1

        #出力層の更新幅計算
        dWkj:np.ndarray = np.zeros(self.Wkj.shape)
        for k in range(K):
            for j in range(J):
                dWkj[k][j] = ETA * (yk_hat[k] - yk[k])*yk[k]*(1-yk[k])*yj1[j]

        #中間層の更新幅計算
        dWji:np.ndarray = np.zeros(self.Wji.shape)
        for j in range(J):
            for i in range(I):
                tmp = 0
                for k in range(K):
                    tmp += (yk_hat[k] - yk[k])*yk[k]*(1-yk[k])*self.Wkj[k][j] 
                dWji[j][i] = ETA * (1 - yj1[j])*yi[i] * tmp

        #出力層更新
        self.Wkj = self.Wkj + dWkj + ALPHA*self.dWkj_t_1
        self.dWkj_t_1 = dWkj
        #中間層更新
        self.Wji = self.Wji + dWji + ALPHA*self.dWji_t_1
        self.dWji_t_1 = dWji

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
        self.yj1 = np.insert(yj, len(yj), 1)
        u = self.Wkj @ self.yj1
        yk = logistic(u)

        return yk

TRAIN_LIMIT_L2NORM = 0.0001
def trainModel(model:NeuralNet, dataSet:List[dataparser.CharData]):
    P = len(dataSet)
    epoch = 0

    while True:
        epoch += 1
        print("Epock数:",epoch)

        #学習
        for p in range(P):
            model.train(dataSet[p])
        
        #学習結果と誤差の検証
        l2ErrRate = 0
        for p in range(P):
            yk_hat = dataSet[p].ansLabel.ansVec
            yk = model.forward(dataSet[p])
            K = len(yk)
            l2ErrRate += (((np.linalg.norm(yk_hat - yk))**2)/K)/P

        print("平均二乗誤差:", l2ErrRate)
        if l2ErrRate < TRAIN_LIMIT_L2NORM:
            return