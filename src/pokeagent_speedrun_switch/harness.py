from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ALLOWED_KEYS = {
    "A",
    "B",
    "X",
    "Y",
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "START",
    "SELECT",
    "WAIT",
    "A_UNTIL_END_OF_DIALOG",
}

SERIAL_KEY_MAP = {
    "A_UNTIL_END_OF_DIALOG": "A",
}


HARNESS_PROMPT = """
You are an autonomous Pokemon FireRed agent playing through a Nintendo Switch video capture.

Your goal is to complete the game efficiently and reliably. You can only see screenshots, recent
action history, persistent memory, and objectives. You do not have RAM data, so screenshots are the
authoritative source for menus, dialogs, battles, and overworld position.

Core priorities:
1. Progress toward becoming Champion.
2. Avoid loops and wasted repeated inputs.
3. Build useful persistent memory for navigation, puzzles, bosses, saves, and mistakes.
4. Keep objectives as a medium-term quest log, not tiny one-step tasks.
5. Prefer actions that visibly change the next frame.

Controls:
- key_press sends one or more keys: A, B, X, Y, UP, DOWN, LEFT, RIGHT, START, SELECT, WAIT.
- A_UNTIL_END_OF_DIALOG means press A repeatedly to advance dialog/text/battle animations.
- Use A_UNTIL_END_OF_DIALOG instead of many individual A presses when text or battle messages are open.
- In overworld, use direction sequences instead of one-tile moves when the path is simple.
- Do not use SELECT unless there is a clear reason.

Visual policy:
- If a clear text box, battle text, or confirmation prompt is visible, advancing with A_UNTIL_END_OF_DIALOG is usually appropriate.
- If no clear text/menu is visible, treat the scene as overworld and prefer movement over A.
- Use A only for a direct interaction when the player is clearly facing an NPC/object/door/item or confirming a highlighted choice.
- If recent screenshots look unchanged after the same input, change strategy.

Memory policy:
- Use memory for persistent knowledge not visible in screenshots: route notes, puzzle findings, save context, boss lessons, menu lessons.
- Use key prefixes such as tips_, route_, puzzle_, boss_, save_, item_, hm_.
- Do not store trivial current-state details that are visible right now.

Objective policy:
- Objectives must explain why the goal matters and how to pursue it.
- Do not write micro-objectives such as "press A" or "move up".

Output JSON only, matching this shape:
{
  "chat_message": "short public commentary",
  "step_details": "why these actions are appropriate now",
  "actions": [
    {"type":"write_memory","key":"tips_example","value":"..."},
    {"type":"update_objectives","primary":{"short_description":"...","description":"..."},
     "secondary":{"short_description":"...","description":"..."},
     "third":{"short_description":"...","description":"..."}},
    {"type":"key_press","keys":["DOWN","DOWN","RIGHT"]}
  ]
}

Rules:
- Always include at least one key_press action unless you are only repairing invalid memory/objectives after an explicit error.
- Keep key_press sequences short enough to recover if wrong: usually 1-8 keys, longer only in open areas.
- A_UNTIL_END_OF_DIALOG may be the only key in a key_press action.
- Never output markdown fences or extra text outside JSON.
""".strip()


@dataclass
class HarnessConfig:
    data_dir: Path = Path("gpt_data")
    history_limit: int = 40
    recent_frames_in_prompt: int = 3
    jpeg_quality: int = 70
    dialog_a_presses: int = 6
    inter_key_delay_sec: float = 0.08
    dialog_key_delay_sec: float = 0.18


@dataclass
class HarnessState:
    memory: dict[str, str] = field(default_factory=dict)
    objectives: dict[str, Any] = field(default_factory=dict)
    markers: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=lambda: {"current_step": 0})


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        backup = path.with_suffix(path.suffix + f".bad-{int(time.time())}")
        path.replace(backup)
        return default


def load_state(config: HarnessConfig) -> HarnessState:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    return HarnessState(
        memory=_read_json(config.data_dir / "memory.json", {}),
        objectives=_read_json(config.data_dir / "objectives.json", {}),
        markers=_read_json(config.data_dir / "markers.json", []),
        history=_read_json(config.data_dir / "history.json", []),
        counters=_read_json(config.data_dir / "counters.json", {"current_step": 0}),
    )


def save_state(config: HarnessConfig, state: HarnessState) -> None:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    (config.data_dir / "memory.json").write_text(
        json.dumps(state.memory, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (config.data_dir / "objectives.json").write_text(
        json.dumps(state.objectives, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (config.data_dir / "markers.json").write_text(
        json.dumps(state.markers, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (config.data_dir / "history.json").write_text(
        json.dumps(state.history[-config.history_limit :], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (config.data_dir / "counters.json").write_text(
        json.dumps(state.counters, ensure_ascii=False, indent=2), encoding="utf-8"
    )


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


def crop_dialog_area(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return frame[int(h * 0.60) : int(h * 0.99), int(w * 0.02) : int(w * 0.98)]


def parse_json_safely(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"JSON parse failed: {text}") from None
        data = json.loads(stripped[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("model output must be a JSON object")
    return data


def build_user_content(
    frames: list[np.ndarray],
    state: HarnessState,
    config: HarnessConfig,
    detail: str,
) -> list[dict[str, Any]]:
    recent_history = state.history[-12:]
    text = {
        "current_step": int(state.counters.get("current_step", 0)),
        "memory": state.memory,
        "objectives": state.objectives,
        "markers": state.markers,
        "recent_history": recent_history,
    }
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": "<harness_state>\n"
            + json.dumps(text, ensure_ascii=False, indent=2)
            + "\n</harness_state>",
        }
    ]

    selected = frames[-config.recent_frames_in_prompt :]
    for idx, frame in enumerate(selected, start=1):
        content.append({"type": "input_text", "text": f"frame_{idx}: full screen"})
        content.append(
            {
                "type": "input_image",
                "image_url": to_data_url_bgr(frame, config.jpeg_quality),
                "detail": detail,
            }
        )
        content.append({"type": "input_text", "text": f"frame_{idx}: bottom dialog/menu area"})
        content.append(
            {
                "type": "input_image",
                "image_url": to_data_url_bgr(crop_dialog_area(frame), config.jpeg_quality),
                "detail": detail,
            }
        )
    return content


def normalize_key(key: Any) -> str:
    normalized = str(key).strip().upper()
    if normalized == "A_UNTIL_END_OF_DIALOG":
        return normalized
    return normalized if normalized in ALLOWED_KEYS else "WAIT"


def normalize_decision(data: dict[str, Any]) -> dict[str, Any]:
    actions = data.get("actions")
    if not isinstance(actions, list):
        single = data.get("action", "WAIT")
        actions = [{"type": "key_press", "keys": [single]}]

    normalized_actions: list[dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        action_type = str(action.get("type", "")).strip()
        if action_type == "key_press":
            raw_keys = action.get("keys", [])
            if isinstance(raw_keys, str):
                raw_keys = [raw_keys]
            if not isinstance(raw_keys, list):
                raw_keys = ["WAIT"]
            keys = [normalize_key(k) for k in raw_keys]
            if keys:
                normalized_actions.append({"type": "key_press", "keys": keys})
        elif action_type in {"write_memory", "delete_memory", "update_objectives", "add_marker", "delete_marker"}:
            normalized_actions.append(action)

    if not any(a.get("type") == "key_press" for a in normalized_actions):
        normalized_actions.append({"type": "key_press", "keys": ["WAIT"]})

    return {
        "chat_message": str(data.get("chat_message", "")).strip(),
        "step_details": str(data.get("step_details", "")).strip(),
        "actions": normalized_actions,
    }


def apply_metadata_actions(state: HarnessState, actions: list[dict[str, Any]]) -> None:
    for action in actions:
        action_type = action.get("type")
        if action_type == "write_memory":
            key = str(action.get("key", "")).strip()
            value = str(action.get("value", "")).strip()
            if key and value:
                state.memory[key] = value
        elif action_type == "delete_memory":
            key = str(action.get("key", "")).strip()
            if key:
                state.memory.pop(key, None)
        elif action_type == "update_objectives":
            state.objectives = {
                "primary": action.get("primary", {}),
                "secondary": action.get("secondary", {}),
                "third": action.get("third", {}),
            }
        elif action_type == "add_marker":
            marker = {
                "label": str(action.get("label", "")).strip(),
                "description": str(action.get("description", action.get("label", ""))).strip(),
                "created_step": int(state.counters.get("current_step", 0)),
            }
            if marker["label"]:
                state.markers.append(marker)
        elif action_type == "delete_marker":
            label = str(action.get("label", "")).strip()
            if label:
                state.markers = [m for m in state.markers if m.get("label") != label]


def key_sequence_from_actions(actions: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    for action in actions:
        if action.get("type") != "key_press":
            continue
        for key in action.get("keys", []):
            keys.append(normalize_key(key))
    return keys or ["WAIT"]
