# Fatigue Monitor AI (MFTF)

Fatigue Monitor AI は、Webカメラによる顔認識とキーボード・マウスの操作量に基づいて、ユーザーの疲労度をリアルタイムに推定・可視化するモダンなWebアプリケーションです。

## 特徴
- **Monitor View**: ライブカメラとリアルタイムチャートで現在の疲労度を監視。
- **History View**: 過去の作業セッションを自動記録。統計（平均疲労度、アラート回数、作業時間）とセッション一覧を表示。
- **Settings View**: 疲労度の閾値、アラート音量、インターフェースのアクセントカラーを自由にカスタマイズ可能。設定は永続化されます。
- **インテリジェント予測**: 直近のデータから60秒後の疲労度を予測し、未然に休憩を促します。
- **Awayモード**: 離席時にモニタリングを一時停止し、統計の正確性を維持。
- **デザイン**: グラスモーフィズムを採用した高級感のあるダークモードUI。

## 技術スタック
- **Frontend**: 
  - [Vue.js 3](https://vuejs.org/) (CDN) - リアクティブなUI管理
  - [Chart.js](https://www.chartjs.org/) - 疲労度推移の描画
  - Vanilla CSS - カスタムデザイン、グラスモーフィズム
- **Backend**:
  - [FastAPI](https://fastapi.tiangolo.com/) - 非同期通信、WebSocket、APIエンドポイント
  - [OpenCV](https://opencv.org/) & [MediaPipe](https://mediapipe.dev/) - 顔認識、姿勢推定（Pitch）
  - [SQLAlchemy](https://www.sqlalchemy.org/) & [aiosqlite](https://aiosqlite.omnilib.dev/en/stable/) - 非同期DB操作、セッション保存
  - [scikit-learn](https://scikit-learn.org/) - 線形回帰を用いた将来予測
  - [pynput](https://pynput.readthedocs.io/) - マウス・キーボード活動量の記録

## フォルダ構成
```text
MFTF/
├── backend/            # FastAPI バックエンドサーバー
│   └── main.py         # サーバーロジック、DBマッピング、WebSocket、API
├── frontend/           # フロントエンドリソース
│   ├── index.html      # シングルページUI構造
│   ├── app.js          # Vue.js アプリケーション・ロジック、チャート制御、API連携
│   └── style.css       # デザイン・スタイル、テーマ色管理
├── db/                 # データベース・ディレクトリ
│   └── fatigue_history.db # 永続化された計測データ (SQLite)
├── resources/          # 音声ファイル（アラート音、BGM）
├── .env                # 環境設定ファイル
├── requirements.txt    # 依存Pythonライブラリ
└── README.md
```

## セットアップと実行

### 1. 前提条件
- Python 3.10以上推奨
- Webカメラが接続されていること

### 2. 環境構築
仮想環境を作成し、必要なライブラリをインストールします。
```bash
# 仮想環境の作成と有効化
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存ライブラリのインストール
pip install -r requirements.txt
```

### 3. アプリケーションの実行
以下のコマンドでサーバーを起動します。
```bash
uvicorn backend.main:app --reload
```
起動後、ブラウザで [http://localhost:8000](http://localhost:8000) にアクセスしてください。

## 詳細機能
### セッションの保存
監視中に「Take a Break」ボタンを押す、またはアプリケーションを終了すると、その間の計測データが自動的にサマリーとして `History` に保存されます。

### 設定のカスタマイズ
`Settings` 画面から以下の項目を変更できます：
- **Fatigue Threshold**: アラートを出す感度を調整。
- **Audio**: 通知音量。
- **Appearance**:UIのアクセントカラー（Cyan, Purple, Green）を切り替え。

## 開発・保守
- **プロセスの終了**: ウィンドウを閉じると、バックエンドはセッションを保存し、リソース（カメラ、入力監視スレッド）を自動的に解放します。
- **データのリセット**: `Settings > Data Management` から過去の全ての履歴を削除可能です。