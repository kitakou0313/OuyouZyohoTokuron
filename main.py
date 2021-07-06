from os import PathLike
from utils import dataparser

if __name__ == "__main__":
    inputDatas = dataparser.parseInputData("data/hira0_00L.dat")
    for data in inputDatas:
        for line in data.data:
            print(line)