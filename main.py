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
        ETA=0.2,
        ALPHA=0.9,
        INPUT_LAYER_WIDTH=64,
        MIDDLE_LAYER_WIDTH=100,
        OUTPUT_LAYER_WIDTH=20
    )
    modelTrainedWriter0 = neuralNet.NeuralNet(nnParams)

    TRAIN_LIMIT_L2NORM = 0.045
    neuralNet.trainModel(modelTrainedWriter0, writer0TrainDatas[::10], TRAIN_LIMIT_L2NORM)

    print("--------------------")
    print("2.1で学習したニューラルネットに筆記者0の学習用データを入力して識別を行なえ。")
    neuralNet.validateModel(modelTrainedWriter0, writer0TrainDatas)
