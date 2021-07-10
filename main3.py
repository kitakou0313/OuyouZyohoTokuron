from utils import dataparser
from network import neuralNet
from typing import List

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

    trainData = writer0TrainDatas + writer1TrainDatas
    testData = writer0TestDatas + writer1TestDatas
    print("データ読み込み完了")
    print("--------------------")
    print("9.筆記者0と筆記者1の学習用データを用いて、ニューラルネットの学習を行なえ。")
    nnParams = neuralNet.NNParams(
        ETA=0.2,
        ALPHA=0.7,
        INPUT_LAYER_WIDTH=64,
        MIDDLE_LAYER_WIDTH=150,
        OUTPUT_LAYER_WIDTH=20
    )
    modelTrainedWriter0AndWriter1 = neuralNet.NeuralNet(nnParams)

    TRAIN_LIMIT_L2NORM_DIFF = 0.001
    neuralNet.trainModel(modelTrainedWriter0AndWriter1, trainData, TRAIN_LIMIT_L2NORM_DIFF)

    print("--------------------")
    print("10.9で学習したニューラルネットに筆記者0と筆記者1の学習用データを入力して識別を行なえ。")
    neuralNet.validateModel(modelTrainedWriter0AndWriter1, trainData)

    print("--------------------")
    print("11.9で学習したニューラルネットに筆記者0と筆記者1のテスト用データを入力して識別を行なえ。")
    neuralNet.validateModel(modelTrainedWriter0AndWriter1, testData)
    print("\n")