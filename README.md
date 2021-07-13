# OuyouZyohoTokuron
応用情報特論（2021）の課題リポジトリ

## mainの種類
- 実験の1~4... `main.py`
- 実験の5~8... `main2.py`
- 実験の9~11... `main3.py`

で分割してあります。

##  実験の実行
### 実験環境の構築

1. `pip install -r requirements.txt`で依存モジュールをインストール可能です。
2. `data`ディレクトリを作成し、そこに各文字のファイルを配置します。
3. makeコマンドで実験可能です。

- 実験1~4... `make exec1`
- 実験5~8... `make exec2`
- 実験9~11... `make exec3`

##  実験結果
`logs`ディレクトリに保存してあります。`numpy.ramdom`のseedが固定されているため、対象のログの実験を再現可能です。

