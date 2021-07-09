class NeuralNet(object):
    """
    NeuralNetworkクラス

    入力:64次元ベクトル（メッシュ特徴の次元数）
    中間層:適当
    出力:20次元ベクトル（文字の種類と同数）

    """
    def __init__(self, numOfMiddleLayers:int) -> None:
        self.layer = []

    def train(self) -> None:
        pass