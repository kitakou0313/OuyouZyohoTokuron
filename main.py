from utils import dataparser
from typing import List

DATAPATH = "data/"
if __name__ == "__main__":
    writer1TrainDatas:List[dataparser.CharData] = []

    for dataNum in range(0, 20):
        path = DATAPATH + "hira0_" + str(dataNum).zfill(2) + "L.dat"
        writer1TrainDatas += dataparser.parseInputData(path)

    writer1TestDatas:List[dataparser.CharData] = []

    for dataNum in range(0, 20):
        path = DATAPATH + "hira0_" + str(dataNum).zfill(2) + "T.dat"
        writer1TestDatas += dataparser.parseInputData(path)

    writer2TrainDatas:List[dataparser.CharData] = []

    for dataNum in range(0, 20):
        path = DATAPATH + "hira1_" + str(dataNum).zfill(2) + "L.dat"
        writer2TrainDatas += dataparser.parseInputData(path)

    writer2TestDatas:List[dataparser.CharData] = []

    for dataNum in range(0, 20):
        path = DATAPATH + "hira1_" + str(dataNum).zfill(2) + "T.dat"
        writer2TestDatas += dataparser.parseInputData(path)

    pass