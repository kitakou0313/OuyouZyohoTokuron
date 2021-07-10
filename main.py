from utils import dataparser
from network import neuralNet
from typing import List
import numpy as np

INPUT_LAYER_WIDTH = 64

DATAPATH = "data/"
if __name__ == "__main__":
    print("データの読み込み")

    print("筆者0:train")
    writer0TrainDatas:List[dataparser.CharData] = []
    for dataNum in range(0, 20):
        path = DATAPATH + "hira0_" + str(dataNum).zfill(2) + "L.dat"
        writer0TrainDatas += dataparser.parseInputData(path,dataNum)

    print("筆者0:test")
    writer0TestDatas:List[dataparser.CharData] = []
    for dataNum in range(0, 20):
        path = DATAPATH + "hira0_" + str(dataNum).zfill(2) + "T.dat"
        writer0TestDatas += dataparser.parseInputData(path,dataNum)

    print("筆者1:test")
    writer1TestDatas:List[dataparser.CharData] = []
    for dataNum in range(0, 20):
        path = DATAPATH + "hira1_" + str(dataNum).zfill(2) + "T.dat"
        writer1TestDatas += dataparser.parseInputData(path,dataNum)

    print("データ読み込み完了")
    print("--------------------")
    print("1.筆記者0の学習用データを用いて、ニューラルネットの学習を行なえ。")
    nnParams = neuralNet.NNParams(
        ETA=0.01,
        ALPHA=0.1,
        INPUT_LAYER_WIDTH=64,
        MIDDLE_LAYER_WIDTH=150,
        OUTPUT_LAYER_WIDTH=20
    )

    np.random.seed(seed=20210711)
    modelTrainedWriter0 = neuralNet.NeuralNet(nnParams)

    TRAIN_LIMIT_L2NORM_DIFF = 0.0001
    neuralNet.trainModel(modelTrainedWriter0, writer0TrainDatas[::50], TRAIN_LIMIT_L2NORM_DIFF)

    print("--------------------")
    print("2.1で学習したニューラルネットに筆記者0の学習用データを入力して識別を行なえ。")
    neuralNet.validateModel(modelTrainedWriter0, writer0TrainDatas)

    print("--------------------")
    print("3.1で学習したニューラルネットに筆記者0のテスト用データを入力して識別を行なえ。")
    neuralNet.validateModel(modelTrainedWriter0, writer0TestDatas)
    print("\n")

    print("--------------------")
    print("4.1で学習したニューラルネットに筆記者1のテスト用データを入力して識別を行なえ。")
    neuralNet.validateModel(modelTrainedWriter0, writer1TestDatas)
    print("\n")