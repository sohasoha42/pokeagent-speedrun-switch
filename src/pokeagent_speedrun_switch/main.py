import os
import cv2
import time
import json
import base64
import argparse
import serial
import numpy as np
from collections import deque
from dotenv import load_dotenv
from openai import OpenAI
from serial.tools import list_ports

load_dotenv()


SYSTEM_PROMPT = """
You are a single-step gameplay agent for Pokemon FireRed.

You receive multiple screenshots in chronological order (oldest -> newest).
Your job is to choose exactly one next button press for the newest screenshot.

Allowed actions:
A, B, X, Y, UP, DOWN, LEFT, RIGHT, START, SELECT

Return format:
{"action":"A","reason":"overworld_move"}

Output rules:
- Return JSON only.
- Return exactly one JSON object.
- action must be exactly one allowed action.
- reason must be one of these fixed labels only:
dialog, confirm, menu_move, battle_move, overworld_move, interact, back, unstick, open_menu

Core rule:
- A is NOT the default action.
- If the screen is unclear, prefer movement instead of A.
- Only choose A when there is strong visual evidence that A will immediately progress the game state.

Mandatory visual check before choosing A:
Choose A only if at least one of these is clearly visible in the newest image:
1. an active text box
2. a visible choice prompt already ready to confirm
3. battle text that is clearly waiting to advance
4. the player character is directly facing a person or object to interact with

If none of the above is clearly visible, do NOT choose A.

Anti-loop rule:
- Compare the newest image with earlier images.
- If the scene looks very similar across frames, assume the previous action may have failed or did not help.
- Do NOT repeat A on similar-looking frames unless the text content or prompt visibly changed.
- If A was likely correct but the scene still looks unchanged, switch to another action that is more likely to create a state change.

Dialogue detection rule:
- Do NOT call it dialogue unless a text box is clearly visible.
- Do NOT infer dialogue from vague dark boxes, borders, or UI-like shapes.
- If no clear text box is visible, it is not dialogue.

Scene policy:

1. Dialogue
- If a clear text box is visible and is waiting to advance, choose A with reason "dialog".
- If a choice menu is visible and the correct option is already highlighted, choose A with reason "confirm".
- If a choice menu is visible but the cursor is not on the desired option, choose UP or DOWN with reason "menu_move".

2. Menu
- Use UP or DOWN to move the cursor with reason "menu_move".
- Use A only if the highlighted option should clearly be confirmed, with reason "confirm".
- Use B to exit only if backing out is clearly better, with reason "back".

3. Battle
- If battle text is clearly open, choose A with reason "dialog".
- If command selection is needed, use direction buttons with reason "battle_move".
- Use A only when the desired battle option is already highlighted, with reason "confirm".

4. Overworld
- If there is no clear text box or menu, treat the scene as overworld.
- In overworld, prefer movement over A.
- Move toward exits, doors, stairs, NPCs, item balls, or open walkable paths.
- If the player seems blocked or stuck across frames, try a different direction with reason "unstick".
- If directly facing an NPC or object, choose A with reason "interact".

5. Unclear scene
- If uncertain whether the screen is dialogue or overworld, treat it as overworld.
- In unclear scenes, prefer UP/DOWN/LEFT/RIGHT over A.

Button restrictions:
- SELECT should almost never be used.
- X and Y should almost never be used.
- START should only be used when opening the menu is clearly needed.

Reason mapping:
- dialog: clear active text to advance
- confirm: confirm highlighted option
- menu_move: move menu cursor
- battle_move: move battle cursor
- overworld_move: move in the field
- interact: talk or inspect
- back: cancel or exit menu
- unstick: try a different direction after no progress
- open_menu: open main menu

Final priority:
- Prefer the action most likely to visibly change the next frame.
- If choosing between A and movement in a non-obvious scene, choose movement.
""".strip()


def to_data_url_bgr(frame: np.ndarray, jpeg_quality: int = 70) -> str:
    ok, buf = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
    )
    if not ok:
        raise RuntimeError("JPEG encode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def crop_regions(frame: np.ndarray):
    """
    1枚の元フレームから、
    - 全体縮小
    - 下部ダイアログ帯
    の2種類を作る
    """
    h, w = frame.shape[:2]

    # 全体
    full = frame

    # 下部帯（雑に会話欄狙い）
    y1 = int(h * 0.62)
    y2 = int(h * 0.98)
    x1 = int(w * 0.03)
    x2 = int(w * 0.97)
    bottom = frame[y1:y2, x1:x2]

    return full, bottom


def parse_json_safely(text: str):
    text = text.strip()

    # そのままJSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # 前後に余計な文字がある場合に備える
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass

    raise ValueError(f"JSON parse failed: {text}")



def find_first_serial_port() -> str:
    """
    接続されている最初のCOMポートを返す。
    FT232だけを挿しておく運用ならこれで十分。
    """
    ports = list(list_ports.comports())
    if not ports:
        raise RuntimeError("COMポートが見つかりません。FT232がPCに挿さっているか確認してください。")
    return ports[0].device


def open_serial_port(port: str | None, baud: int):
    """
    port が未指定なら最初に見つかったCOMポートを使う。
    """
    if port is None or port.lower() == "auto":
        port = find_first_serial_port()

    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(2)  # FT232/シリアル接続の安定待ち
    print(f"serial opened: {port} @ {baud}", flush=True)
    return ser


def send_hid_action(ser, action: str):
    """
    Picoへ 1行コマンドとして action を送る。
    例: A\n, UP\n, START\n
    """
    action = str(action).upper().strip()
    ser.write((action + "\n").encode("utf-8"))
    ser.flush()


def call_vlm(client: OpenAI, model: str, frames: list[np.ndarray], detail: str = "low"):
    """
    frames は古い→新しい順
    各時点について:
      - full frame
      - bottom crop
    を送る
    """
    content = [
        {
            "type": "input_text",
            "text": SYSTEM_PROMPT,
        }
    ]

    for idx, frame in enumerate(frames, start=1):
        full, bottom = crop_regions(frame)

        content.append({
            "type": "input_text",
            "text": f"frame_{idx}: full screen"
        })
        content.append({
            "type": "input_image",
            "image_url": to_data_url_bgr(full),
            "detail": detail,
        })

        content.append({
            "type": "input_text",
            "text": f"frame_{idx}: bottom dialog area"
        })
        content.append({
            "type": "input_image",
            "image_url": to_data_url_bgr(bottom),
            "detail": detail,
        })

    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": "あなたは厳密にJSONのみを返します。"}
                ],
            },
            {
                "role": "user",
                "content": content,
            }
        ],
    )

    text = resp.output_text
    data = parse_json_safely(text)

    action = str(data.get("action", "WAIT")).upper()
    reason = str(data.get("reason", ""))

    allowed = {"A", "B", "X", "Y", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT", "WAIT"}
    if action not in allowed:
        action = "WAIT"

    return {
        "action": action,
        "reason": reason,
        "raw": text,
    }


def open_capture(camera_index: int, width: int, height: int):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"camera_index={camera_index} を開けませんでした")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def draw_overlay(frame: np.ndarray, status_lines: list[str]):
    y = 30
    for line in status_lines:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--sample-every-sec", type=float, default=0.7)
    parser.add_argument("--num-frames", type=int, default=3)
    parser.add_argument("--model", type=str, default="gpt-5.4-mini")
    parser.add_argument("--detail", type=str, default="low", choices=["low", "high", "auto"])
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--serial", action="store_true", help="VLMのactionをシリアル通信でPicoへ送信する")
    parser.add_argument("--serial-port", type=str, default="auto", help="例: COM5。autoなら最初に見つかったCOMポート")
    parser.add_argument("--serial-baud", type=int, default=115200)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が設定されていません。.env を確認してください。")

    client = OpenAI(api_key=api_key)
    cap = open_capture(args.camera_index, args.width, args.height)
    ser = open_serial_port(args.serial_port, args.serial_baud) if args.serial else None

    frame_buffer = deque(maxlen=args.num_frames)
    last_sample_ts = 0.0
    last_decision_ts = 0.0
    last_result = {"action": "-", "reason": "collecting frames"}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("frame read failed")
                time.sleep(0.05)
                continue

            now = time.time()

            if now - last_sample_ts >= args.sample_every_sec:
                frame_buffer.append(frame.copy())
                last_sample_ts = now

            # バッファが埋まったらVLMへ問い合わせ
            if len(frame_buffer) == args.num_frames and (now - last_decision_ts >= args.sample_every_sec):
                try:
                    result = call_vlm(
                        client=client,
                        model=args.model,
                        frames=list(frame_buffer),
                        detail=args.detail,
                    )

                    changed = (
                        result["action"] != last_result["action"]
                        or result["reason"] != last_result["reason"]
                    )
                    last_result = result
                    last_decision_ts = now

                    if ser is not None:
                        send_hid_action(ser, result["action"])
                        print(
                            f'sent HID: {result["action"]} | reason: {result["reason"]}',
                            flush=True
                        )
                    elif changed:
                        print(
                            f'suggested: {result["action"]} | reason: {result["reason"]}',
                            flush=True
                        )

                except Exception as e:
                    print(f"vlm error: {e}", flush=True)
                    last_result = {"action": "WAIT", "reason": f"error: {e}"}
                    last_decision_ts = now

            if args.show:
                preview = frame.copy()
                draw_overlay(
                    preview,
                    [
                        f"model: {args.model}",
                        f"frames buffered: {len(frame_buffer)}/{args.num_frames}",
                        f"suggested HID: {last_result['action']}",
                        f"reason: {last_result['reason']}",
                        "ESC: quit",
                    ],
                )
                cv2.imshow("pokemon-vlm-hid-suggester", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break

    finally:
        if ser is not None:
            ser.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()