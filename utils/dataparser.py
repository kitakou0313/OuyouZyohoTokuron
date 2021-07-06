from typing import List

class CharData():
    """
    入力用のデータ型
    """
    def __init__(self, data:List[List[int]]) -> None:
        self.data:List[List[int]] = data

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
