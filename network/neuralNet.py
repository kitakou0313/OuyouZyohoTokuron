from typing import List
import numpy as np
from numpy.random.mtrand import rand
from utils import dataparser
from tqdm import tqdm

def logistic(x):
    y = 1 / (1 + np.exp(-x))
    return y

def softmax(x):
    x_max = x.max(axis=0)
    x = x - x_max
    w = np.exp(x)
    return (w / w.sum(axis=0))

class NNParams():
    """
    ネットワークのパラメータ管理用クラス
    """
    def __init__(self, ETA:int, ALPHA:int,INPUT_LAYER_WIDTH:int,MIDDLE_LAYER_WIDTH:int, OUTPUT_LAYER_WIDTH:int) -> None:
        self.ETA = ETA
        self.ALPHA = ALPHA
        self.INPUT_LAYER_WIDTH = INPUT_LAYER_WIDTH
        self.MIDDLE_LAYER_WIDTH = MIDDLE_LAYER_WIDTH
        self.OUTPUT_LAYER_WIDTH = OUTPUT_LAYER_WIDTH

class NeuralNet(object):
    """
    NeuralNetworkクラス
    入力:64次元ベクトル（メッシュ特徴の次元数）
    中間層:定数で指定
    出力:20次元ベクトル（文字の種類と同数）
    """
    def __init__(self, params:NNParams) -> None:
        self.params = params

        #標準分布（平均0、標準偏差1）で初期化
        self.Wji:np.ndarray = np.random.randn(self.params.MIDDLE_LAYER_WIDTH,self.params.INPUT_LAYER_WIDTH)
        self.Wkj:np.ndarray = np.random.randn(self.params.OUTPUT_LAYER_WIDTH,self.params.MIDDLE_LAYER_WIDTH)
        
        #学習時の重み更新用の中間層の出力
        self.yj:np.ndarray = np.array([])

        #学習安定用の前回学習時の更新幅
        self.dWji_t_1:np.ndarray = np.zeros(self.Wji.shape)
        self.dWkj_t_1:np.ndarray = np.zeros(self.Wkj.shape)
    def train(self,charData:dataparser.CharData) -> None:
        """
        学習用関数
        """
        yk = self.forward(charData)
        yj = self.yj
        yi = charData.meshFeature
        yk_hat = charData.ansLabel.ansVec

        #出力層の更新幅計算
        dWkj:np.ndarray = self.params.ETA * ((yk_hat - yk).reshape(-1, 1))*yk.reshape(-1, 1)*(1-yk.reshape(-1, 1))*yj

        #中間層の更新幅計算
        tmp = np.sum((yk_hat - yk)*yk*(1-yk)*self.Wkj.T, axis=1)
        dWji:np.ndarray = self.params.ETA*tmp.reshape(-1, 1)*yj.reshape(-1,1)*((1 - yj).reshape(-1,1))*yi


        #出力層更新
        self.Wkj = self.Wkj + dWkj + self.params.ALPHA*self.dWkj_t_1
        self.dWkj_t_1 = dWkj
        #中間層更新
        self.Wji = self.Wji + dWji + self.params.ALPHA*self.dWji_t_1
        self.dWji_t_1 = dWji

    def forward(self, charData:dataparser.CharData) -> np.ndarray:
        """
        識別用関数
        """
        inputVector = charData.meshFeature

        #中間層
        inputVectorWith1 = inputVector
        u = self.Wji @ inputVectorWith1
        yj = logistic(u)

        #出力層
        self.yj = yj
        u = self.Wkj @ self.yj
        yk = softmax(u)

        return yk

def trainModel(model:NeuralNet, dataSet:List[dataparser.CharData], TRAIN_LIMIT_L2NORM:float):
    P = len(dataSet)
    epoch = 0
    preStepErrRate = 0

    while True:
        epoch += 1
        print("Epoch数:",epoch)

        #学習
        for p in tqdm(range(P)):
            model.train(dataSet[p])
        
        #学習結果と誤差の検証
        l2ErrRate = 0
        for p in range(P):
            yk_hat = dataSet[p].ansLabel.ansVec
            yk = model.forward(dataSet[p])
            K = len(yk)
            l2ErrRate += (((np.linalg.norm(yk_hat - yk))**2)/K)/P

        print("平均二乗誤差:", l2ErrRate)
        print("改善された誤差:", preStepErrRate - l2ErrRate, "\n")
        #誤差が事前に指定したより小さくなれば停止
        if l2ErrRate < TRAIN_LIMIT_L2NORM:
            return
        
        preStepErrRate = l2ErrRate

def validateModel(model:NeuralNet, dataSet:List[dataparser.CharData]):
    correctNum = 0 
    for dataInd in range(len(dataSet)):
        charData = dataSet[dataInd]
        res = model.forward(charData)
        resMoziType = np.argmax(res)

        if dataInd % 100 == 0:
            print("ベクトル")
            print(res)

            print("正解:", charData.ansLabel.charType, "予測:", dataparser.CHAR_IND[resMoziType])

            if resMoziType == charData.ansLabel.charInd:
                print("正解！")
            else:
                print("不正解")

        if resMoziType == charData.ansLabel.charInd:
            correctNum += 1
    
    print("正答率:", int( (correctNum/len(dataSet))*100 ), "%")