from typing import List
import numpy as np
from numpy.random.mtrand import rand
from utils import dataparser
from tqdm import tqdm

def logistic(x):
    y = 1 / (1 + np.exp(-x))
    return y

def softmax(x):
    x = x.T
    x_max = x.max(axis=0)
    x = x - x_max
    w = np.exp(x)
    return (w / w.sum(axis=0)).T

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
    入力:64次元ベクトル（メッシュ特徴の次元数）+ 1次元(1で固定)
    中間層:定数で指定
    出力:20次元ベクトル（文字の種類と同数）
    """
    def __init__(self, params:NNParams) -> None:
        self.params = params

        #定数1を追加する用の次元を用意する
        inputLayerWidthWith1 = self.params.INPUT_LAYER_WIDTH + 1
        middleLayerWidth1With1 = self.params.MIDDLE_LAYER_WIDTH + 1

        #標準偏差（平均0、標準偏差1）で初期化
        self.Wji:np.ndarray = np.random.randn(self.params.MIDDLE_LAYER_WIDTH,inputLayerWidthWith1)
        self.Wkj:np.ndarray = np.random.randn(self.params.OUTPUT_LAYER_WIDTH,middleLayerWidth1With1)
        
        #0.5で初期化
        """
        self.Wji:np.ndarray = np.full((self.params.MIDDLE_LAYER_WIDTH,inputLayerWidthWith1), 0.5)
        self.Wkj:np.ndarray = np.full((self.params.OUTPUT_LAYER_WIDTH,middleLayerWidth1With1), 0.5)
        """
        
        #学習時の重み更新用の中間層の出力
        self.yj1:np.ndarray = np.array([])

        #学習安定用の前回学習時の更新幅
        self.dWji_t_1:np.ndarray = np.zeros(self.Wji.shape)
        self.dWkj_t_1:np.ndarray = np.zeros(self.Wkj.shape)
    @profile
    def train(self,charData:dataparser.CharData) -> None:
        """
        学習用関数
        """
        yk = self.forward(charData)
        yj1 = self.yj1
        yi = np.insert(charData.meshFeature, len(charData.meshFeature), 1)
        yk_hat = charData.ansLabel.ansVec

        K = self.params.OUTPUT_LAYER_WIDTH
        J1 = self.params.MIDDLE_LAYER_WIDTH + 1
        J = self.params.MIDDLE_LAYER_WIDTH
        I = self.params.INPUT_LAYER_WIDTH + 1

        #出力層の更新幅計算
        dWkj:np.ndarray = np.zeros(self.Wkj.shape)
        for k in range(K):
            """
            for j in range(J1):
            """
            dWkj[k] = self.params.ETA * (yk_hat[k] - yk[k])*yk[k]*(1-yk[k])*yj1


        #中間層の更新幅計算
        dWji:np.ndarray = np.zeros(self.Wji.shape)
        for j in range(J):
            tmp = np.sum((yk_hat - yk)*yk*(1-yk)*self.Wkj[:,j])
            dWji[j] = self.params.ETA *yj1[j]*(1 - yj1[j])*yi * tmp


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
        inputVectorWith1 = np.insert(inputVector, len(inputVector), 1)
        u = self.Wji @ inputVectorWith1
        yj = logistic(u)

        #出力層
        self.yj1 = np.insert(yj, len(yj), 1)
        u = self.Wkj @ self.yj1
        yk = softmax(u)

        return yk
@profile
def trainModel(model:NeuralNet, dataSet:List[dataparser.CharData], TRAIN_LIMIT_L2NORM_DIFF:float):
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
        #誤差が事前に指定した幅から更新されなくなったら収束として停止
        if abs(preStepErrRate - l2ErrRate) < TRAIN_LIMIT_L2NORM_DIFF:
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