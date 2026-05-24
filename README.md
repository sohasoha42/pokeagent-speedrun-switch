# pokeagent-speedrun-switch

OpenAI の視覚モデルに Nintendo Switch のキャプチャ映像を見せて、次に押すボタンを決めさせる Python 製の実験ハーネスです。

現在のプロンプトは Pokemon FireRed の speedrun / playthrough を想定しています。OpenCV で映像を読み取り、OpenAI Responses API へ直近フレームを送り、返ってきた JSON からキー入力を取り出します。シリアル接続の入力ブリッジがある場合は、判断されたキーを実機へ送信できます。

## 構成

```text
.
├── scripts/
│   ├── detect_cameras.py   # 利用可能なカメラ番号を確認する
│   ├── test_openai.py      # OpenAI API 接続を確認する
│   └── test_serial.py      # 最初に見つかった COM ポートへ A を送る
├── firmware/
│   └── serial-hid-bridge/  # シリアル受信側マイコンコード
├── src/pokeagent_speedrun_switch/
│   ├── main.py             # キャプチャ、API 呼び出し、入力送信の実行ループ
│   └── harness.py          # プロンプト、状態保存、JSON 正規化、画像変換
├── .env.example
├── pyproject.toml
└── uv.lock
```

## 必要なもの

- Python 3.14 以上
- uv
- OpenAI API キー
- Nintendo Switch の映像を PC へ取り込めるキャプチャ環境
- 実入力を送る場合は、シリアル経由でキーを受け取るコントローラブリッジ

## セットアップ

```powershell
git clone https://github.com/sohasoha42/pokeagent-speedrun-switch.git
cd pokeagent-speedrun-switch
Copy-Item .env.example .env
uv sync
```

`.env` に API キーを入れます。

```env
OPENAI_API_KEY=your_api_key_here
```

必要ならモデルと reasoning effort も環境変数で指定できます。

```env
OPENAI_MODEL=gpt-5.4-mini
OPENAI_REASONING_EFFORT=medium
```

## 起動

通常はキャプチャデバイスを自動検出します。検出がうまくいかない場合は、カメラ番号を確認します。

```powershell
uv run python scripts/detect_cameras.py
```

入力を送信せず、判断内容だけを確認する場合:

```powershell
uv run pokeagent-speedrun-switch --dry-run
```

シリアル入力ブリッジへ送信する場合:

```powershell
uv run pokeagent-speedrun-switch --serial --serial-port COM3
```

COM ポートを自動検出する場合:

```powershell
uv run pokeagent-speedrun-switch --serial --serial-port auto
```

## 実行時の流れ

1. OpenCV がキャプチャデバイスからフレームを取得します。
2. 指定間隔で直近フレームをためます。
3. フルスクリーンのフレーム、履歴、メモリ、目標を OpenAI Responses API へ送ります。
4. モデルは `chat_message`、`step_details`、`actions` を含む JSON を返します。
5. `actions` のうち `key_press` がキー列へ変換されます。
6. `--serial` が有効ならシリアルポートへ送信します。無効または `--dry-run` なら標準出力へ表示します。
7. 判断履歴と状態が `gpt_data/` に保存されます。

## 主なオプション

| オプション | 既定値 | 内容 |
| --- | --- | --- |
| `--camera-index` | `auto` | OpenCV で開くカメラ番号。`auto` なら利用可能なカメラを自動検出 |
| `--width` | `1280` | キャプチャ幅 |
| `--height` | `720` | キャプチャ高さ |
| `--sample-every-sec` | `0.5` | フレームをバッファへ追加する間隔 |
| `--decision-every-sec` | `1.0` | モデルへ判断を求める間隔 |
| `--num-frames` | `3` | 1 回の判断で使う直近フレーム数 |
| `--model` | `OPENAI_MODEL` または `gpt-5.4-mini` | 使用する OpenAI モデル |
| `--reasoning-effort` | `OPENAI_REASONING_EFFORT` または `medium` | reasoning effort |
| `--detail` | `low` | 画像入力の detail。`low`、`high`、`auto` |
| `--serial` | 無効 | シリアルポートへ入力を送信 |
| `--dry-run` | 無効 | 入力を送信せず、送信予定のキーだけ表示 |
| `--serial-port` | `auto` | 使用する COM ポート |
| `--serial-baud` | `115200` | シリアル通信速度 |
| `--data-dir` | `gpt_data` | 状態ファイルの保存先 |
| `--dpad-turn-hold-sec` | `0.08` | Python 側で送る方向合わせ用の短い保持秒数 |
| `--dpad-walk-hold-sec` | `1.0` | Python 側で送る歩行用の保持秒数。既定では `UP` を `UP:80`, `UP:1000` として送信 |

## モデルが使えるキー

`key_press` のキーは次の値に正規化されます。

```text
A
B
X
Y
UP
DOWN
LEFT
RIGHT
START
SELECT
WAIT
A_UNTIL_END_OF_DIALOG
```

許可されていないキーは `WAIT` に置き換えられます。`A_UNTIL_END_OF_DIALOG` はダイアログや戦闘メッセージを進める意図を表す論理キーで、HID 送信時は `A` 1 回に変換されます。

## 状態保存

既定では `gpt_data/` に以下のファイルが作られます。

| ファイル | 内容 |
| --- | --- |
| `memory.json` | 画面だけでは分からない永続メモ |
| `objectives.json` | 現在の中期目標 |
| `markers.json` | モデルが追加した進行マーカー |
| `history.json` | 直近の判断、キー、エラー |
| `counters.json` | ステップ番号 |

新しい試行として始めたい場合は、プログラムを終了してから `gpt_data/` を別名にするか削除します。

## 動作確認

OpenAI API:

```powershell
uv run python scripts/test_openai.py
```

シリアル:

```powershell
uv run python scripts/test_serial.py
```

`test_serial.py` は最初に見つかった COM ポートへ `A` を送信します。接続先を確認してから実行してください。

## 注意

- API 利用料金が発生します。
- `--serial` を付けると実際に入力が送られます。最初は `--dry-run` で確認してください。
- モデルの画面認識や判断は失敗することがあります。
- キャプチャデバイス番号と COM ポートは環境によって変わります。
- このリポジトリはゲームプレイ自動化の実験用です。利用するゲーム、機器、サービスの規約に従ってください。

## License

MIT License
