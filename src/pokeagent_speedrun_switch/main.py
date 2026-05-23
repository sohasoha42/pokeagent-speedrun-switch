from __future__ import annotations

import argparse
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import serial
from dotenv import load_dotenv
from openai import OpenAI
from serial.tools import list_ports

from .camera import open_camera, usable_camera_indices
from .harness import (
    HARNESS_PROMPT,
    HarnessConfig,
    HarnessState,
    MOVEMENT_KEYS,
    apply_metadata_actions,
    build_visual_change_summary,
    build_user_content,
    guard_against_reinteraction_loop,
    guard_against_stagnation,
    key_sequence_from_actions,
    load_state,
    normalize_decision,
    parse_json_safely,
    save_state,
)

load_dotenv()


def find_first_serial_port() -> str:
    ports = list(list_ports.comports())
    if not ports:
        raise RuntimeError("No COM ports found. Connect the controller bridge and try again.")
    return ports[0].device


def open_serial_port(port: str | None, baud: int) -> serial.Serial:
    if port is None or port.lower() == "auto":
        port = find_first_serial_port()
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(2)
    print(f"serial opened: {port} @ {baud}", flush=True)
    return ser


def send_hid_key(ser: serial.Serial, key: str) -> None:
    ser.write((key.upper().strip() + "\n").encode("utf-8"))
    ser.flush()


def serial_commands_for_key(key: str, config: HarnessConfig) -> list[str]:
    if key in MOVEMENT_KEYS:
        turn_hold_ms = int(config.dpad_turn_hold_sec * 1000)
        step_hold_ms = int(config.dpad_step_hold_sec * config.dpad_steps_per_move * 1000)
        return [f"{key}:{turn_hold_ms}", f"{key}:{step_hold_ms}"]
    return [key]


def execute_keys(
    ser: serial.Serial | None,
    keys: list[str],
    config: HarnessConfig,
    dry_run: bool,
) -> None:
    for key in keys:
        if key == "WAIT":
            time.sleep(config.inter_key_delay_sec)
            continue

        if key == "A_UNTIL_END_OF_DIALOG":
            for _ in range(config.dialog_a_presses):
                if dry_run or ser is None:
                    print("would send HID: A", flush=True)
                else:
                    send_hid_key(ser, "A")
                time.sleep(config.dialog_key_delay_sec)
            continue

        if dry_run or ser is None:
            for command in serial_commands_for_key(key, config):
                print(f"would send HID: {command}", flush=True)
        else:
            for command in serial_commands_for_key(key, config):
                send_hid_key(ser, command)
        time.sleep(config.inter_key_delay_sec)


def open_capture(camera_index: str, width: int, height: int) -> cv2.VideoCapture:
    if camera_index.lower() != "auto":
        try:
            index = int(camera_index)
        except ValueError:
            raise RuntimeError("--camera-index must be an integer or 'auto'") from None

        cap = open_camera(index, width, height)
        if cap is None:
            raise RuntimeError(f"Could not open camera_index={index}")
        print(f"camera opened: index {index}", flush=True)
        return cap

    indices = usable_camera_indices()
    if indices:
        index = indices[0]
        cap = open_camera(index, width, height)
        if cap is not None:
            print(f"camera opened: index {index} (available: {indices})", flush=True)
            return cap

    raise RuntimeError("No usable camera found. Try --camera-index with a known device index.")


def response_text(resp: Any) -> str:
    text = getattr(resp, "output_text", None)
    if text:
        return str(text)
    chunks: list[str] = []
    for item in getattr(resp, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                chunks.append(str(value))
    return "".join(chunks)


def call_agent(
    client: OpenAI,
    model: str,
    frames: list[np.ndarray],
    state: HarnessState,
    config: HarnessConfig,
    detail: str,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": HARNESS_PROMPT}],
            },
            {
                "role": "user",
                "content": build_user_content(frames, state, config, detail),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "pokemon_harness_decision",
                "schema": {
                    "type": "object",
                    "properties": {
                        "scene_type": {
                            "type": "string",
                            "enum": ["overworld", "dialog", "menu", "battle", "transition", "unclear"],
                        },
                        "chat_message": {"type": "string"},
                        "step_details": {"type": "string"},
                        "actions": {
                            "type": "array",
                            "items": {"type": "object", "additionalProperties": True},
                        },
                    },
                    "required": ["scene_type", "chat_message", "step_details", "actions"],
                    "additionalProperties": False,
                },
                "strict": False,
            }
        },
    }
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}

    resp = client.responses.create(**kwargs)
    return normalize_decision(parse_json_safely(response_text(resp)))


def record_step(
    state: HarnessState,
    decision: dict[str, Any],
    keys: list[str],
    visual_summary: dict[str, Any],
    error: str | None = None,
) -> None:
    state.history.append(
        {
            "step": int(state.counters.get("current_step", 0)),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scene_type": decision.get("scene_type", "unclear"),
            "chat_message": decision.get("chat_message", ""),
            "step_details": decision.get("step_details", ""),
            "actions": decision.get("actions", []),
            "keys": keys,
            "visual_change_summary": visual_summary,
            "error": error,
        }
    )
    state.counters["current_step"] = int(state.counters.get("current_step", 0)) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=str, default="auto")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--sample-every-sec", type=float, default=0.5)
    parser.add_argument("--decision-every-sec", type=float, default=1.0)
    parser.add_argument("--num-frames", type=int, default=3)
    parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--reasoning-effort", type=str, default=os.getenv("OPENAI_REASONING_EFFORT", "medium"))
    parser.add_argument("--detail", type=str, default="low", choices=["low", "high", "auto"])
    parser.add_argument("--serial", action="store_true", help="Send selected keys to the serial controller bridge.")
    parser.add_argument("--dry-run", action="store_true", help="Plan actions but do not send serial inputs.")
    parser.add_argument("--serial-port", type=str, default="auto")
    parser.add_argument("--serial-baud", type=int, default=115200)
    parser.add_argument("--data-dir", type=Path, default=Path("gpt_data"))
    parser.add_argument("--dialog-a-presses", type=int, default=6)
    parser.add_argument("--dpad-turn-hold-sec", type=float, default=0.08)
    parser.add_argument("--dpad-step-hold-sec", type=float, default=0.38)
    parser.add_argument("--dpad-steps-per-move", type=int, default=1)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env or the environment.")

    harness_config = HarnessConfig(
        data_dir=args.data_dir,
        dialog_a_presses=args.dialog_a_presses,
        recent_frames_in_prompt=args.num_frames,
        dpad_turn_hold_sec=args.dpad_turn_hold_sec,
        dpad_step_hold_sec=args.dpad_step_hold_sec,
        dpad_steps_per_move=args.dpad_steps_per_move,
    )
    state = load_state(harness_config)
    client = OpenAI(api_key=api_key)
    cap = open_capture(args.camera_index, args.width, args.height)
    ser = open_serial_port(args.serial_port, args.serial_baud) if args.serial and not args.dry_run else None

    frame_buffer: deque[np.ndarray] = deque(maxlen=args.num_frames)
    last_sample_ts = 0.0
    last_decision_ts = 0.0
    last_status = "collecting frames"
    last_keys: list[str] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("frame read failed", flush=True)
                time.sleep(0.05)
                continue

            now = time.time()
            if now - last_sample_ts >= args.sample_every_sec:
                frame_buffer.append(frame.copy())
                last_sample_ts = now

            ready = len(frame_buffer) == args.num_frames
            due = now - last_decision_ts >= args.decision_every_sec
            if ready and due:
                visual_summary = build_visual_change_summary(list(frame_buffer))
                try:
                    decision = call_agent(
                        client=client,
                        model=args.model,
                        frames=list(frame_buffer),
                        state=state,
                        config=harness_config,
                        detail=args.detail,
                        reasoning_effort=args.reasoning_effort or None,
                    )
                    apply_metadata_actions(state, decision["actions"])
                    keys = key_sequence_from_actions(decision["actions"])
                    keys, stagnation_note = guard_against_stagnation(state, keys, visual_summary)
                    keys, guard_note = guard_against_reinteraction_loop(state, decision, keys)
                    guard_notes = [note for note in [stagnation_note, guard_note] if note]
                    if guard_notes:
                        decision["step_details"] = (
                            (decision.get("step_details") or "").rstrip()
                            + " ["
                            + " | ".join(guard_notes)
                            + "]"
                        ).strip()
                    execute_keys(ser, keys, harness_config, args.dry_run or not args.serial)
                    record_step(state, decision, keys, visual_summary)
                    save_state(harness_config, state)
                    last_keys = keys
                    last_status = decision.get("step_details") or decision.get("chat_message") or "acted"
                    print(
                        f"step {state.counters['current_step']}: keys={keys} | {last_status}",
                        flush=True,
                    )
                except Exception as exc:
                    fallback = {
                        "scene_type": "unclear",
                        "chat_message": "",
                        "step_details": f"agent error: {exc}",
                        "actions": [{"type": "key_press", "keys": ["WAIT"]}],
                    }
                    record_step(state, fallback, ["WAIT"], visual_summary, error=str(exc))
                    save_state(harness_config, state)
                    last_status = f"error: {exc}"
                    print(last_status, flush=True)
                finally:
                    last_decision_ts = now
    finally:
        if ser is not None:
            ser.close()
        cap.release()


if __name__ == "__main__":
    main()
