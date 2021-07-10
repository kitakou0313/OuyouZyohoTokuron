from typing import List
import unittest
import numpy as np

CHAR_IND = {
    0:"あ",
    1:"い",
    2:"う",
    3:"え",
    4:"お",
    5:"か",
    6:"き",
    7:"く",
    8:"け",
    9:"こ",
    10:"さ",
    11:"し",
    12:"す",
    13:"せ",
    14:"そ",
    15:"た",
    16:"ち",
    17:"つ",
    18:"て",
    19:"と"
}

class AnsLabel():
    """
    文字種、データのインデックス、教師用onehotベクトルなどを管理
    """
    def __init__(self, charInd:int) -> None:
        self.charInd:int = charInd
        self.charType:str = CHAR_IND[charInd]
        self.ansVec = [1 if ind == charInd else 0 for ind in range(0, len(CHAR_IND))]

class CharData():
    """
    入力用のデータ型
    """
    def __init__(self, data:List[List[int]], charInd:int) -> None:
        self.data:List[List[int]] = data
        self.meshFeature:np.ndarray = np.array(convertRawDataToMeshFeature(data))
        self.ansLabel:AnsLabel = AnsLabel(charInd)

def parseInputData(path:str,charInd:int) -> List[CharData]:
    """
    パスのファイルのデータ読み込み
    """
    rawData:List[List[int]] = []

    f = open(path, "r")
    for line in f:
        rawData.append(list(map(int, list(line.rstrip('\n')))))

    parsedDatas:List[CharData] = []

    for dataInd in range(0, len(rawData), 64):
        parsedDatas.append(CharData(rawData[dataInd:dataInd+64], charInd))

    return parsedDatas

def convertRawDataToMeshFeature(rawData:List[List[int]]) -> List[int]:
    Y = len(rawData)
    X = len(rawData[0])
    sampleSize = 8

    meshFeature:List[int] = []
    for y in range(0, Y, sampleSize):
        for x in range(0, X, sampleSize):
            tmpSum = 0
            for dy in range(0, sampleSize):
                tmpSum += sum(rawData[y+dy][x:x+sampleSize])
            meshFeature.append(tmpSum /(sampleSize**2))

    return meshFeature

class Test(unittest.TestCase):
    def testMeshFeatureConverter(self):
        testCases = [
            (
                [
                    [0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0],
                    [1,1,1,1,0,0,0,0],
                    [0,0,0,0,1,1,1,1],
                    [0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0],
                ], [0.125]
            ),
            (
                [
                    [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
                    [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
                    [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
                    [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
                    [1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1],
                    [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
                    [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
                    [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
                ], [0.125, 1.0]
            )
        ]

        for testRawData, expected in testCases:
            self.assertEqual(convertRawDataToMeshFeature(testRawData), expected)

if __name__ == "__main__":
    unittest.main()