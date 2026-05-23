# serial-hid-bridge

PC 側の `pokeagent-speedrun-switch --serial` から送られる 1 行コマンドを Raspberry Pi Pico で受信し、Nintendo Switch 用の USB HID 入力へ変換するファームウェアです。

このスケッチは [sorasen2020/SwitchControllerPico](https://github.com/sorasen2020/SwitchControllerPico) に依存しています。

## 接続

Pico の USB は Nintendo Switch に接続し、Switch コントローラーとして認識させます。

PC からのシリアルコマンドは USB-Serial アダプタなどを使って Pico の UART に接続します。

| USB-Serial adapter | Raspberry Pi Pico |
| --- | --- |
| TX | GP1 / UART0 RX |
| RX | GP0 / UART0 TX |
| GND | GND |

既定の UART 設定:

```text
Serial1
baud: 115200
TX: GP0
RX: GP1
line ending: \n
```

## 受信コマンド

PC 側からは以下のコマンドを改行区切りで送ります。

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
```

マイコン側では次のように `SwitchControllerPico` へ変換します。

| Command | SwitchControllerPico |
| --- | --- |
| `A` | `Button::A` |
| `B` | `Button::B` |
| `X` | `Button::X` |
| `Y` | `Button::Y` |
| `UP` | `Hat::UP` |
| `DOWN` | `Hat::DOWN` |
| `LEFT` | `Hat::LEFT` |
| `RIGHT` | `Hat::RIGHT` |
| `START` | `Button::PLUS` |
| `SELECT` | `Button::MINUS` |

`WAIT` と `A_UNTIL_END_OF_DIALOG` も受信できるようにしていますが、通常は PC 側で処理されます。

- `WAIT`: マイコン側で短時間待機
- `A_UNTIL_END_OF_DIALOG`: `A` を 6 回送信

## ビルド前提

- Raspberry Pi Pico
- Arduino IDE または Arduino CLI
- Earle F. Philhower の Raspberry Pi Pico Arduino core
- Adafruit TinyUSB stack
- `SwitchControllerPico`

`SwitchControllerPico` のセットアップ手順に従って、Pico が Switch にコントローラーとして認識される状態にしてください。

## 動作確認

Pico を Switch に接続し、PC から USB-Serial アダプタ経由で以下を送信します。

```text
A
DOWN
START
```

正常に処理した場合、UART 側へ以下のような応答を返します。

```text
OK A
OK DOWN
OK START
```

不明なコマンドは `ERR <command>` を返し、Switch 側入力は送信しません。
