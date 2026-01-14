# MFTF (Fatigue Monitor System)

MFTF (Multimodal Fatigue Tracking Framework) は、Webカメラによる顔認識とキーボード・マウスの操作量に基づいて、ユーザーの疲労度をリアルタイムに推定・可視化するシステムです。

## 特徴
- **リアルタイム疲労度モニタリング**: Webカメラ映像（顔の姿勢）と入力デバイスの活動量から疲労度を算出。
- **グラフ可視化**: 疲労度の推移をグラフで表示。
- **予測機能**: 直近の傾向から未来の疲労度を予測。
- **休憩通知**: 疲労度が一定を超えると休憩を促します。
- **Awayモード**: 離席時にモニタリングを一時停止するモード。

## 技術スタック
- **Backend**: Python (FastAPI), OpenCV, MediaPipe, Pynput
- **Frontend**: HTML5, CSS3, Vue.js (CDN), Chart.js
- **OS**: Linux / Windows (WSL2対応)

## フォルダ構成
```text
MFTF/
├── backend/            # FastAPI バックエンドサーバー
│   ├── main.py         # アプリケーションエントリーポイント
│   └── ...
├── frontend/           # フロントエンドリソース
│   ├── index.html      # メインUI
│   ├── app.js          # Vue.js アプリケーションロジック
│   └── style.css       # スタイルシート
├── resources/          # 音声ファイル等のリソース
├── .env                # 環境設定ファイル
├── requirements.txt    # 依存Pythonライブラリ
└── README.md
```

## セットアップと実行

### 1. 前提条件
- Python 3.8以上
- Webカメラが接続されていること

### 2. 環境構築

#### Linux / macOS の場合
ターミナルで以下を実行します。
```bash
# 仮想環境の有効化
source venv/bin/activate

# 依存ライブラリのインストール
pip install -r requirements.txt
```

#### Windows の場合
コマンドプロンプトまたはPowerShellで以下を実行します。
```cmd
:: 仮想環境の有効化
venv\Scripts\activate

:: 依存ライブラリのインストール
pip install -r requirements.txt
```
※ **注意**: `pynput` (キーボード監視ライブラリ) がウイルス対策ソフトに検知される場合があります。その場合は例外設定に追加してください。

### 3. 設定
必要に応じて `.env` ファイルを編集してパラメータを調整してください。
- `CAMERA_ID`: 使用するカメラのID (デフォルト: 0)
- `FATIGUE_THRESHOLD`: アラートを出す疲労度の閾値 (0-100)

### 4. アプリケーションの実行
以下のコマンドでサーバーを起動します。

```bash
uvicorn backend.main:app --reload
```

### 5. 利用開始
サーバー起動後、ブラウザで以下のURLにアクセスしてください。

[http://localhost:8000](http://localhost:8000)

※ 初回アクセス時にカメラの使用許可を求められますので、許可してください。

## トラブルシューティング
- **カメラが起動しない場合**: ブラウザのカメラ権限設定を確認してください。
- **入力監視エラー**: Linux環境等で `pynput` の権限エラーが出る場合がありますが、アプリケーションの動作（Webインターフェース）には影響しないよう設計されています。