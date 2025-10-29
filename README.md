# MFTF

## フォルダ構成
MFTF/
│
├── src/
│   ├── main.py                # エントリーポイント
│   ├── config/                # 設定ファイル類
│   │   └── settings.py
│   │
│   ├── data/                  # 入出力データ管理
│   │   ├── collector/         # 各種データ取得モジュール
│   │   │   ├── keyboard_monitor.py
│   │   │   ├── mouse_monitor.py
│   │   │   └── webcam_monitor.py
│   │   ├── preprocessor/      # データ整形
│   │   │   ├── signal_filter.py
│   │   │   └── feature_extractor.py
│   │   └── storage/           # データ保存
│   │       └── database.py
│   │
│   ├── model/                 # 機械学習モデル
│   │   ├── fatigue_regressor.py
│   │   └── trainer.py
│   │
│   ├── analysis/              # 可視化・評価
│   │   ├── visualization.py
│   │   └── metrics.py
│   │
│   ├── interface/             # ユーザーインタフェース
│   │   ├── gui.py
│   │   └── notifier.py        # 休憩促し通知など
│   │
│   └── utils/                 # 汎用関数
│       ├── logger.py
│       └── timer.py
│
├── models/                    # 学習済みモデル格納場所
├── data/                      # 生データ保存用
├── requirements.txt
├── README.md
└──.vscode                     # vscode設定

## 使用ライブラリ例と目的
| 機能              | ライブラリ                                 | 主な用途                    |
| --------------- | ------------------------------------- | ----------------------- |
| **キーボード/マウス監視** | `pynput`                              | 入力イベントの記録（クリック数・キータイプ数） |
| **Webカメラ解析**    | `opencv-python`, `mediapipe`          | 顔検出・目の開閉状態・姿勢分析         |
| **データ処理**       | `numpy`, `pandas`                     | 集計、前処理、特徴抽出             |
| **機械学習**        | `scikit-learn`, `xgboost`             | 疲労度予測モデル（回帰）            |
| **可視化**         | `matplotlib`, `seaborn`               | 分析・評価プロット               |
| **通知・GUI**      | `tkinter` / `PyQt5` / `customtkinter` | 休憩促しやステータス表示            |
| **ログ管理**        | `loguru` / `logging`                  | ログ・例外処理                 |
| **構成管理**        | `pydantic` / `yaml`                   | 設定ファイル・モデルパラメータ管理       |

## データフロー
[Keyboard/Mouse] → ActivityCollector
        ↓
   [FeatureExtractor]（1分ごとに特徴量を生成）
        ↓
   [WebcamMonitor]（顔・目・頭部姿勢などを抽出）
        ↓
   [FatigueRegressor]（回帰モデルによる疲労度推定）
        ↓
   [Notifier]（休憩が必要ならGUIや通知を発信）

## 疲労度モデル

# 動作手順
・仮想環境がない場合は、仮想環境を作成する
>> conda create -n DevApp

作成後、環境を有効化
>> conda activate DevApp

・必要なライブラリをインストールする
>> cd MFTF
>> pip install -r requirements.txt

・以下のコマンドを実行して動かしてみる
>> python -m src.main