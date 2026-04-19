import os
import cv2
import time
import json
import base64
import argparse
import numpy as np
from collections import deque
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


SYSTEM_PROMPT = """
あなたはポケモンのプレイ補助です。
入力される複数枚の画像は時系列順です（古い→新しい）。

役割:
- 画面を見て、次に押すべき入力を1つだけ決める
- 候補は次の中から必ず1つだけ選ぶ:
  A, B, UP, DOWN, LEFT, RIGHT, WAIT

判断ルール:
- 会話テキスト送り中なら A を優先
- 暗転・遷移中・ロード中なら WAIT
- 明確なメニューキャンセルが必要そうなら B
- フィールド移動中は方向入力を選ぶ
- 不確実なときは WAIT を選ぶ

必ずJSONのみで返してください。説明文やコードブロックは禁止。
形式:
{"action":"A","reason":"dialog visible"}
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

    allowed = {"A", "B", "UP", "DOWN", "LEFT", "RIGHT", "WAIT"}
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
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が設定されていません。.env を確認してください。")

    client = OpenAI(api_key=api_key)
    cap = open_capture(args.camera_index, args.width, args.height)

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

                    if changed:
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
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()