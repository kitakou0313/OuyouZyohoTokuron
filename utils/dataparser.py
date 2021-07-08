from typing import List
import unittest

class CharData():
    """
    入力用のデータ型
    """
    def __init__(self, data:List[List[int]]) -> None:
        self.data:List[List[int]] = data
        self.meshFeature:List[int] = convertRawDataToMeshFeature(data)

def parseInputData(path:str) -> List[CharData]:
    """
    パスのファイルのデータ読み込み
    """
    rawData:List[List[int]] = []

    f = open(path, "r")
    for line in f:
        rawData.append(list(map(int, list(line.rstrip('\n')))))

    parsedDatas:List[CharData] = []

    for dataInd in range(0, len(rawData), 64):
        parsedDatas.append(CharData(rawData[dataInd:dataInd+64]))

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