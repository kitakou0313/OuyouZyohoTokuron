from utils import dataparser
from network import neuralNet
from typing import List
import numpy as np

DATAPATH = "data/"
if __name__ == "__main__":
    print("データの読み込み")

    print("筆者0:test")
    writer0TestDatas:List[dataparser.CharData] = []
    for dataNum in range(0, 20):
        path = DATAPATH + "hira0_" + str(dataNum).zfill(2) + "T.dat"
        writer0TestDatas += dataparser.parseInputData(path,dataNum)

    print("筆者1:train")
    writer1TrainDatas:List[dataparser.CharData] = []
    for dataNum in range(0, 20):
        path = DATAPATH + "hira1_" + str(dataNum).zfill(2) + "L.dat"
        writer1TrainDatas += dataparser.parseInputData(path,dataNum)

    print("筆者1:test")
    writer1TestDatas:List[dataparser.CharData] = []
    for dataNum in range(0, 20):
        path = DATAPATH + "hira1_" + str(dataNum).zfill(2) + "T.dat"
        writer1TestDatas += dataparser.parseInputData(path,dataNum)

    print("データ読み込み完了")
    print("--------------------")
    print("5.筆記者1の学習用データを用いて、ニューラルネットの学習を行なえ。")
    nnParams = neuralNet.NNParams(
        ETA=0.03,
        ALPHA=0.1,
        INPUT_LAYER_WIDTH=64,
        MIDDLE_LAYER_WIDTH=150,
        OUTPUT_LAYER_WIDTH=20
    )

    np.random.seed(seed=20210713)
    modelTrainedWriter1 = neuralNet.NeuralNet(nnParams)

    TRAIN_LIMIT_L2NORM = 0.0001
    neuralNet.trainModel(modelTrainedWriter1, writer1TrainDatas, TRAIN_LIMIT_L2NORM)
    
    print("--------------------")
    print("6.5で学習したニューラルネットに筆記者1の学習用データを入力して識別を行なえ。")
    neuralNet.validateModel(modelTrainedWriter1, writer1TrainDatas)

    print("--------------------")
    print("7.5で学習したニューラルネットに筆記者0のテスト用データを入力して識別を行なえ。")
    neuralNet.validateModel(modelTrainedWriter1, writer0TestDatas)
    print("\n")

    print("--------------------")
    print("8.5で学習したニューラルネットに筆記者1のテスト用データを入力して識別を行なえ。")
    neuralNet.validateModel(modelTrainedWriter1, writer1TestDatas)
    print("\n")