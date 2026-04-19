# pokeagent-speedrun-switch

Nintendo Switch 上の **『ポケットモンスター ファイアレッド・リーフグリーン』** の画面をキャプチャし、  
OpenAI のモデルに「次に押す入力」を提案させる実験用ツールです。

現在は入力の自動送信は行わず、推奨アクションを表示するだけです。

## できること

- キャプチャ映像を取得する
- 複数フレームをモデルに送る
- 次の入力を提案する

提案される入力は以下です。

- `A`
- `B`
- `UP`
- `DOWN`
- `LEFT`
- `RIGHT`
- `WAIT`

## 動作環境

- Python 3.14+
- OpenAI API キー
- キャプチャデバイス
- Nintendo Switch の映像を取り込める環境

## セットアップ

```bash
git clone https://github.com/sohasoha42/pokeagent-speedrun-switch.git
cd pokeagent-speedrun-switch
cp .env.example .env
```

`.env` に API キーを設定します。

```env
OPENAI_API_KEY=your_api_key_here
```

依存関係をインストールします。

```bash
uv sync
```

## 使い方

```bash
uv run pokeagent-speedrun-switch --show
```

## 主なオプション

- `--camera-index` : カメラ番号
- `--width` : 幅
- `--height` : 高さ
- `--sample-every-sec` : フレーム取得間隔
- `--num-frames` : 送信するフレーム数
- `--model` : 使用モデル
- `--detail` : 画像 detail
- `--show` : プレビュー表示

## 補助スクリプト

カメラ確認:

```bash
uv run python scripts/detect_cameras.py
```

API 確認:

```bash
uv run python scripts/test_openai.py
```

## 注意

- 実験用ツールです
- 実際のコントローラ入力は行いません
- API 利用料金が発生します
- 判断に迷う場合は `WAIT` を返します

## ライセンス

MIT License
