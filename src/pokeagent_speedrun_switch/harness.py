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

MAX_KEYS_PER_KEY_PRESS = 3
MOVEMENT_KEYS = {"UP", "DOWN", "LEFT", "RIGHT"}
A_LIKE_KEYS = {"A", "A_UNTIL_END_OF_DIALOG"}

SCENE_TYPES = {"overworld", "dialog", "menu", "battle", "transition", "unclear"}
OBJECT_INTERACTION_TERMS = (
    "bedroom",
    "furniture",
    "object",
    "pc",
    "tv",
    "snes",
    "famicom",
    "console",
    "game console",
    "sign",
    "decoration",
    "bookshelf",
    "same object",
)
STORY_DIALOG_TERMS = (
    "professor oak",
    "oak",
    "mom",
    "rival",
    "name",
    "naming",
    "keyboard",
    "confirmation",
    "choice",
    "yes/no",
    "battle text",
)
LEDGE_TERMS = (
    "ledge",
    "brown ledge",
    "horizontal ledge",
    "cliff",
    "ridge",
    "段差",
)
UPWARD_BLOCK_TERMS = (
    "blocked",
    "stalled",
    "stalling",
    "stuck",
    "cannot climb",
    "can't climb",
    "not climbable",
    "under the ledge",
    "below the ledge",
    "direct north path",
    "straight up",
    "straight north",
    "north lane",
    "upward path",
)
LATERAL_LEDGE_PROGRESS_TERMS = (
    "opening",
    "gap",
    "corridor",
    "open lane",
    "right-hand lane",
    "left-hand lane",
    "making visible progress",
    "visibly advancing",
    "visibly shifted",
    "visibly moving",
)
LATERAL_LEDGE_STOP_TERMS = (
    "tree wall",
    "tree line",
    "map edge",
    "edge of the map",
    "edge of route",
    "far edge",
    "corner",
    "wall",
    "pinned",
    "stalled",
    "stuck",
    "blocked",
    "cramped",
    "hard stop",
    "dead end",
    "no visible movement",
    "did not move",
    "does not move",
    "can't move",
    "cannot move",
    "stops moving",
    "stops producing visible movement",
)
LEDGE_OPENING_PROBE_TERMS = (
    "opening",
    "gap",
    "new alignment",
    "new position",
    "from this lane",
    "from this alignment",
    "test up",
    "test a short up",
    "test the opening",
    "upward check",
    "probe north",
)
DOORWAY_LOOP_TERMS = (
    "doorway",
    "door",
    "entrance",
    "frontage",
    "re-trigger",
    "retrigger",
    "oak's lab",
    "lab entrance",
    "lab door",
)

EARLY_GAME_BOOTSTRAP_GUIDE = """
Early-game visual route guide:
- Opening intro/name screens: advance text and prompts normally.
- On player/rival naming screens, do not use START to finish. Enter the desired short name, move the cursor to the on-screen "おわる" / finish option, then press A to confirm.
- If a naming keyboard is open and a name has already been entered, the objective is to navigate to "おわる" and press A, not press START.
- First controllable scene is the player's bedroom. The objective is to leave the room, not inspect furniture.
- In the bedroom, ignore the PC, TV/SNES, signs, and decorations unless a text box is already open.
- Bedroom stairs in FireRed are a dark stair/doorway tile on the room edge, often toward the right/upper-right side of the room. Treat dark stair-like edge tiles as the exit target, not as an NPC/sign/object to inspect.
- Bedroom navigation should use visual landmarks: first locate the player, then identify likely stairs/door tiles on the room edge, then move toward that target.
- If you are in an overworld bedroom-like room with no text box, do not inspect objects. Move toward the most plausible stair/exit tile with a short route such as RIGHT,RIGHT,UP or RIGHT,UP,UP when the exit appears above/right, or another route justified by the newest screenshot.
- If a route does not visibly change position, mark that direction as likely blocked in step_details and try a different route. Do not keep following a hard-coded direction pattern.
- After leaving the bedroom, go downstairs, exit the house, then head north toward Route 1 / Oak's scripted stop.
- After the starter choice and first rival battle are complete, leave Oak's lab and route north toward Route 1. Do not orbit Oak's lab exterior or repeatedly chase Oak outside unless a visible dialog/prompt or a required adjacent interaction is clearly blocking progress.
- Around Oak's lab exterior, if no dialog is open and the current objective is Route 1, treat Oak, signs, and the lab door as things to route away from. Step away from the doorway/frontage first, then find a clean northbound lane out of Pallet Town.
""".strip()


HARNESS_PROMPT = """
You are an autonomous Pokemon FireRed agent playing through a Nintendo Switch video capture.

Your goal is to complete the game efficiently and reliably. You can only see screenshots, recent
action history, persistent memory, and objectives. You do not have RAM data, so screenshots are the
authoritative source for menus, dialogs, battles, and overworld position. Unlike RAM-based harnesses,
you must navigate from visual landmarks and action history.

Core priorities:
1. Progress toward becoming Champion.
2. Avoid loops and wasted repeated inputs.
3. Build useful persistent memory for navigation, puzzles, bosses, saves, and mistakes.
4. Keep objectives as a medium-term quest log, not tiny one-step tasks.
5. Prefer actions that visibly change the next frame.

Controls:
- key_press sends a sequential list of keys: A, B, X, Y, UP, DOWN, LEFT, RIGHT, START, SELECT, WAIT.
- Put at most 3 keys in one key_press action. For longer routes, emit multiple key_press actions in order.
- A_UNTIL_END_OF_DIALOG is a semantic alias for one A press, then re-observe before deciding whether another A is needed.
- Use A_UNTIL_END_OF_DIALOG when text or battle messages are open, but do not rely on it to clear long dialog in one decision.
- In overworld, use direction sequences instead of one-tile moves when the path is simple. Moving 3-8 tiles across multiple key_press actions is often better than dithering.
- On naming keyboards, finish by moving to "おわる" and pressing A. Do not use START as a shortcut for name completion.
- Do not use SELECT unless there is a clear reason.

Visual policy:
- If a clear text box, battle text, or confirmation prompt is visible, advancing with A_UNTIL_END_OF_DIALOG is usually appropriate.
- When latest_game_view_zoom is present, use it as the primary source for tile-level terrain, ledge openings, stairs, doors, and the player tile. Use the full capture for broader context and UI.
- If no clear text/menu is visible, treat the scene as overworld and prefer movement over A.
- Use A only for a direct interaction when the player is directly adjacent to, and clearly facing, an NPC/object/door/item, or when confirming a highlighted choice.
- If the player is not adjacent to the target, or the player's facing direction is uncertain/wrong, move or turn first. Do not include A in the same action unless the final position and facing are clear from the newest screenshot.
- If recent screenshots look unchanged after the same input, change strategy.
- Images are provided in chronological order, oldest to newest. Judge the newest frame, but use older frames to detect whether dialog disappeared, whether movement happened, or whether the scene is stuck.
- Do not assume dialog is still active just because prior frames or history had dialog. The newest frame must visibly contain a text box/menu/battle prompt to classify as dialog/menu/battle.
- If visual_change_summary says the latest frames changed very little after repeated A-like inputs, avoid more A unless a visible continuation arrow or prompt remains in the newest frame.
- If stagnation_summary.is_stagnant is true, deliberately choose a different tactic from recent_history: change movement direction, back out with B if in a menu, or wait only for transitions. Do not repeat the same key sequence.
- Because this harness has no RAM/minimap, use screenshot-based local exploration: infer likely walkable tiles, test a short route, use visual changes to update which directions are blocked, and prefer exits/stairs/doors over interacting with furniture.
- Exception: when the newest frame clearly shows dialog/text/battle text/confirmation, advancing it with A_UNTIL_END_OF_DIALOG or A is progress, not stagnation. Do not avoid A just because recent frames changed little during dialog.

Navigation policy:
- Do not immediately reverse through the same doorway or map transition. After exiting a building, first move laterally or farther away from the entrance before trying to route north/south.
- If recent_history alternates between entering/exiting the same map or between opposite directions, treat that as a route loop. Break it with a side lane, not by repeating the forward/back inputs.
- When an objective says to route north but the player is directly below a door, do not press UP from the doorway frontage. Step sideways out of the doorway column, then resume north.

Terrain policy:
- Brown ledges in Pokemon FireRed are one-way terrain. You can jump down from the high side, but you cannot climb up them from below.
- If the player is below/south of a horizontal brown ledge, do not press UP to try to cross it. Move left/right along the ledge until a real opening is visible, or move away and route around.
- For Route 1 ledges, identify the actual visual gap/opening in the newest game_view_zoom before committing to UP. If the gap is not visible, continue lateral exploration or backtrack to a lower lane instead of guessing from old memory.
- If recent_history shows UP repeatedly stalling at a ledge, mark that tile/lane as blocked and stop re-testing UP from the same side.
- Do not oscillate left and right under the same ledge. Pick one lateral search direction and commit to it for a meaningful distance before reversing.
- A meaningful lateral search is usually at least 4-6 direction presses or 2-3 consecutive lateral decisions in the same direction, unless the newest frame clearly shows a hard stop such as a tree wall, map edge, corner, or no visible movement.
- After a meaningful lateral move changes position, test the opening with UP once; if both lateral sides are blocked, step DOWN/backtrack to a lower lane and route around.
- Do not continue lateral movement indefinitely while describing the lane as open or making progress. After committing far enough in one lateral direction, try UP from the new x-position to check whether you reached the opening.

Interaction loop policy:
- Talking to an NPC, reading a sign, checking an object, or opening a one-shot message is complete once its text box disappears.
- Text from bedroom furniture, the PC, TV/SNES/Famicom, signs, or decorations is one-shot flavor text. After advancing it once, stop pressing A and move away unless a story NPC, menu, naming keyboard, or confirmation prompt is clearly visible.
- If the current screen is overworld and recent history already used A or A_UNTIL_END_OF_DIALOG to talk/check/read, do not press A again while still facing the same person or object.
- After finishing dialog with a person/object, the next overworld action should usually be movement away from that target or toward the route/objective.
- Repeatedly talking to the same target is allowed only when a visible prompt/menu/choice requires confirmation, or when an objective explicitly says repeated interaction is required.
- Overworld A is valid only at interaction range: the player must be on the neighboring tile and facing the target. If there is any gap, diagonal offset, or wrong facing, output only movement/turning first.
- If unsure whether a conversation just ended, choose a short movement input instead of A.

Memory policy:
- Use memory for persistent knowledge not visible in screenshots: route notes, puzzle findings, save context, boss lessons, menu lessons.
- Use key prefixes such as tips_, route_, puzzle_, boss_, save_, item_, hm_.
- Do not store trivial current-state details that are visible right now.

Map policy:
- Use maps and markers as spatial memory for navigation. Add markers for stairs, exits, doors, important NPCs, items, shops, centers, blockers, and failed paths.
- A marker must include map_id, map_name, x, y, emoji, and label. Coordinates are local tile-like estimates; use consistent approximate coordinates within each map even when exact RAM coordinates are unavailable.
- map_id should be a stable lowercase identifier such as player_bedroom, player_house_1f, pallet_town, route_1, or oaks_lab.
- Check existing maps before choosing movement. If a marker describes stairs, exits, or a known blocker, use it to route or avoid repeating mistakes.
- Treat old route memories as hypotheses, not commands. If recent_history shows that a local Oak/lab-frontage alignment has failed repeatedly, stop following older route_oak_* alignment notes and either add/update a blocker marker or route toward the current objective by a different lane.

Objective policy:
- Objectives must explain why the goal matters and how to pursue it.
- Do not write micro-objectives such as "press A" or "move up".

Scene classification:
- Set scene_type to exactly one of: overworld, dialog, menu, battle, transition, unclear.
- If there is no clear text box/menu/battle UI, scene_type must be overworld or transition, not dialog.
- In scene_type=overworld, do not output A or A_UNTIL_END_OF_DIALOG unless an explicit visible interaction target is required for progress.
- If a bedroom/house interior is visible and no text box is open, scene_type is overworld and the action should be movement toward stairs/exits.

Output JSON only, matching this shape:
{
  "scene_type": "overworld",
  "chat_message": "short public commentary",
  "step_details": "why these actions are appropriate now",
  "actions": [
    {"type":"write_memory","key":"tips_example","value":"..."},
    {"type":"add_marker","map_id":"pallet_town","map_name":"Pallet Town","x":5,"y":8,
     "emoji":"door","label":"Player house front door; use this to leave home and route north."},
    {"type":"delete_marker","map_id":"pallet_town","x":5,"y":8},
    {"type":"update_objectives","primary":{"short_description":"...","description":"..."},
     "secondary":{"short_description":"...","description":"..."},
     "third":{"short_description":"...","description":"..."}},
    {"type":"key_press","keys":["DOWN","DOWN","RIGHT"]}
  ]
}

Rules:
- Always include at least one key_press action unless you are only repairing invalid memory/objectives after an explicit error.
- Keep each key_press action short enough to recover if wrong: at most 3 keys per key_press. Use multiple key_press actions for simple overworld routes that need more than 3 movement keys.
- A_UNTIL_END_OF_DIALOG may be the only key in a key_press action.
- Never output markdown fences or extra text outside JSON.
""".strip()


@dataclass
class HarnessConfig:
    data_dir: Path = Path("gpt_data")
    history_limit: int = 40
    recent_frames_in_prompt: int = 3
    jpeg_quality: int = 90
    include_latest_game_view: bool = True
    game_view_min_width: int = 960
    inter_key_delay_sec: float = 0.08
    dpad_turn_hold_sec: float = 0.08
    dpad_walk_hold_sec: float = 1.0


@dataclass
class HarnessState:
    memory: dict[str, str] = field(default_factory=dict)
    objectives: dict[str, Any] = field(default_factory=dict)
    maps: dict[str, Any] = field(default_factory=dict)
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
    maps = _read_json(config.data_dir / "maps.json", {})
    markers = _read_json(config.data_dir / "markers.json", [])
    if not maps and markers:
        maps = build_maps_from_markers(markers)
    return HarnessState(
        memory=_read_json(config.data_dir / "memory.json", {}),
        objectives=_read_json(config.data_dir / "objectives.json", {}),
        maps=maps,
        markers=markers,
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
    (config.data_dir / "maps.json").write_text(
        json.dumps(state.maps, ensure_ascii=False, indent=2), encoding="utf-8"
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


def marker_key(marker: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(marker.get("map_id", "")).strip(),
        str(marker.get("x", "")).strip(),
        str(marker.get("y", "")).strip(),
    )


def marker_position(marker: dict[str, Any]) -> tuple[str, float, float] | None:
    try:
        return (
            str(marker.get("map_id", "")).strip(),
            float(marker.get("x")),
            float(marker.get("y")),
        )
    except (TypeError, ValueError):
        return None


def normalize_marker(action: dict[str, Any], step: int) -> dict[str, Any] | None:
    map_id = str(action.get("map_id", action.get("map_name", "unknown"))).strip()
    map_name = str(action.get("map_name", map_id)).strip()
    label = str(action.get("label", action.get("description", ""))).strip()
    if not map_id or not label:
        return None

    try:
        x = float(action.get("x"))
        y = float(action.get("y"))
    except (TypeError, ValueError):
        return None

    marker = {
        "map_id": map_id,
        "map_name": map_name or map_id,
        "x": x,
        "y": y,
        "emoji": str(action.get("emoji", "marker")).strip() or "marker",
        "label": label,
        "description": str(action.get("description", label)).strip() or label,
        "created_step": int(action.get("created_step", step) or step),
        "updated_step": step,
    }
    return marker


def build_maps_from_markers(markers: list[dict[str, Any]]) -> dict[str, Any]:
    maps: dict[str, Any] = {}
    for marker in markers:
        map_id = str(marker.get("map_id", marker.get("map_name", "unknown"))).strip()
        if not map_id:
            continue
        map_entry = maps.setdefault(
            map_id,
            {
                "map_id": map_id,
                "map_name": str(marker.get("map_name", map_id)).strip() or map_id,
                "markers": [],
            },
        )
        map_entry["markers"].append(marker)
    return maps


def upsert_marker(state: HarnessState, marker: dict[str, Any]) -> None:
    key = marker_key(marker)
    state.markers = [existing for existing in state.markers if marker_key(existing) != key]
    state.markers.append(marker)
    state.maps = build_maps_from_markers(state.markers)


def delete_marker(state: HarnessState, action: dict[str, Any]) -> None:
    map_id = str(action.get("map_id", "")).strip()
    try:
        x = float(action.get("x"))
        y = float(action.get("y"))
    except (TypeError, ValueError):
        return

    state.markers = [
        marker
        for marker in state.markers
        if marker_position(marker) != (map_id, x, y)
    ]
    state.maps = build_maps_from_markers(state.markers)


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


def extract_game_view_bgr(frame: np.ndarray, padding_ratio: float = 0.015) -> np.ndarray | None:
    if frame.size == 0:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Capture cards commonly present the GBA image inside a black border. A low
    # threshold keeps dark caves/interiors intact while still removing the border.
    mask = gray > 8
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None

    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    height, width = frame.shape[:2]
    crop_width = right - left
    crop_height = bottom - top
    if crop_width < width * 0.2 or crop_height < height * 0.2:
        return None
    if crop_width > width * 0.96 and crop_height > height * 0.96:
        return None

    pad_x = max(2, int(crop_width * padding_ratio))
    pad_y = max(2, int(crop_height * padding_ratio))
    left = max(0, left - pad_x)
    right = min(width, right + pad_x)
    top = max(0, top - pad_y)
    bottom = min(height, bottom + pad_y)
    return frame[top:bottom, left:right].copy()


def upscale_for_prompt(frame: np.ndarray, min_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= 0 or width >= min_width:
        return frame
    scale = min_width / width
    target = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(frame, target, interpolation=cv2.INTER_NEAREST)


def frame_difference_ratio(previous: np.ndarray, current: np.ndarray) -> float:
    prev_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    if prev_gray.shape != curr_gray.shape:
        curr_gray = cv2.resize(curr_gray, (prev_gray.shape[1], prev_gray.shape[0]))
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(np.mean(diff) / 255.0)


def build_visual_change_summary(frames: list[np.ndarray]) -> dict[str, Any]:
    selected = frames[-min(len(frames), 5) :]
    diffs = [
        round(frame_difference_ratio(previous, current), 4)
        for previous, current in zip(selected, selected[1:])
    ]
    if not diffs:
        trend = "single_frame_only"
    elif max(diffs) < 0.01:
        trend = "nearly_static"
    elif max(diffs) < 0.04:
        trend = "small_changes"
    else:
        trend = "changed"
    return {
        "frames_available": len(frames),
        "frames_sent": len(frames),
        "oldest_to_newest_diff_ratios": diffs,
        "trend": trend,
    }


def is_low_visual_change(summary: dict[str, Any]) -> bool:
    trend = str(summary.get("trend", ""))
    if trend in {"nearly_static", "single_frame_only"}:
        return True
    diffs = summary.get("oldest_to_newest_diff_ratios", [])
    if not isinstance(diffs, list) or not diffs:
        return False
    numeric_diffs = [float(value) for value in diffs if isinstance(value, int | float)]
    return bool(numeric_diffs) and max(numeric_diffs) < 0.015


def build_stagnation_summary(
    history: list[dict[str, Any]],
    current_visual_summary: dict[str, Any],
    lookback: int = 5,
) -> dict[str, Any]:
    recent = history[-lookback:]
    low_change_entries = [
        entry for entry in recent if is_low_visual_change(entry.get("visual_change_summary", {}))
    ]
    recent_keys = [key for entry in recent for key in _entry_keys(entry)]
    recent_non_wait_keys = [key for key in recent_keys if key != "WAIT"]

    repeated_key = None
    if len(recent_non_wait_keys) >= 3 and len(set(recent_non_wait_keys[-3:])) == 1:
        repeated_key = recent_non_wait_keys[-1]

    a_like_count = sum(1 for key in recent_non_wait_keys if key in A_LIKE_KEYS)
    movement_count = sum(1 for key in recent_non_wait_keys if key in MOVEMENT_KEYS)
    low_change_now = is_low_visual_change(current_visual_summary)
    stagnant = low_change_now and (
        len(low_change_entries) >= 2 or repeated_key is not None or a_like_count >= 2
    )

    return {
        "is_stagnant": stagnant,
        "low_change_recent_steps": len(low_change_entries),
        "low_change_now": low_change_now,
        "repeated_recent_key": repeated_key,
        "recent_a_like_count": a_like_count,
        "recent_movement_count": movement_count,
    }


def build_navigation_summary(history: list[dict[str, Any]], lookback: int = 12) -> dict[str, Any]:
    recent = history[-lookback:]
    movement_sequences = [
        [key for key in _entry_keys(entry) if key in MOVEMENT_KEYS]
        for entry in recent
    ]
    movement_sequences = [keys for keys in movement_sequences if keys]
    flat_moves = [key for keys in movement_sequences for key in keys]

    reverse_pairs = {("UP", "DOWN"), ("DOWN", "UP"), ("LEFT", "RIGHT"), ("RIGHT", "LEFT")}
    immediate_reversals = sum(
        1 for previous, current in zip(flat_moves, flat_moves[1:]) if (previous, current) in reverse_pairs
    )
    vertical_count = sum(1 for key in flat_moves if key in {"UP", "DOWN"})
    horizontal_count = sum(1 for key in flat_moves if key in {"LEFT", "RIGHT"})
    recent_text = " ".join(
        (str(entry.get("chat_message", "")) + " " + str(entry.get("step_details", ""))).lower()
        for entry in recent
    )
    doorway_loop_risk = (
        any(term in recent_text for term in DOORWAY_LOOP_TERMS)
        and "UP" in flat_moves[-8:]
        and "DOWN" in flat_moves[-8:]
    )

    recommendation = ""
    if doorway_loop_risk:
        recommendation = "Break the doorway loop: move sideways out of the entrance column before trying north again."
    elif immediate_reversals >= 3:
        recommendation = "Recent movement is oscillating; choose a side lane or backtrack to a lower lane before retrying forward progress."
    elif vertical_count >= 6 and horizontal_count <= 2:
        recommendation = "Recent movement is mostly vertical; add lateral movement to avoid bouncing between transitions or blockers."

    lateral_commitment = _lateral_commitment(history)
    if lateral_commitment["is_short"]:
        direction = lateral_commitment["direction"]
        recommendation = (
            f"Continue the current {direction} lateral search longer before reversing; "
            "the recent side probe is still too short to conclude that lane failed."
        )

    return {
        "recent_movement_sequences": movement_sequences[-6:],
        "immediate_reversals": immediate_reversals,
        "vertical_count": vertical_count,
        "horizontal_count": horizontal_count,
        "lateral_commitment": lateral_commitment,
        "doorway_loop_risk": doorway_loop_risk,
        "recommendation": recommendation,
    }


def select_memory_for_prompt(
    memory: dict[str, str],
    objectives: dict[str, Any],
    recent_history: list[dict[str, Any]],
    max_entries: int = 80,
) -> dict[str, str]:
    if len(memory) <= max_entries:
        return memory

    context = (
        json.dumps(objectives, ensure_ascii=False)
        + " "
        + json.dumps(recent_history[-8:], ensure_ascii=False)
    ).lower()
    current_terms = {
        "route_1",
        "route 1",
        "ledge",
        "opening",
        "viridian",
        "grass",
        "corridor",
        "north",
        "pallet",
    }
    if "lab" in context or "oak" in context:
        current_terms.update({"lab", "oak", "doorway", "frontage"})

    scored: list[tuple[int, int, str, str]] = []
    items = list(memory.items())
    for index, (key, value) in enumerate(items):
        text = (key + " " + value).lower()
        score = 0
        if any(term in text for term in current_terms):
            score += 4
        if any(term in context and term in text for term in current_terms):
            score += 3
        if key.startswith(("tips_", "boss_", "hm_", "item_", "save_")):
            score += 1
        # Later entries are usually more accurate for local routing because they were learned
        # after older failed probes.
        scored.append((score, index, key, value))

    selected = sorted(scored, key=lambda entry: (entry[0], entry[1]), reverse=True)[:max_entries]
    selected.sort(key=lambda entry: entry[1])
    return {key: value for _, _, key, value in selected}


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
    visual_change_summary = build_visual_change_summary(frames)
    selected_memory = select_memory_for_prompt(state.memory, state.objectives, recent_history)
    selected = frames[-config.recent_frames_in_prompt :]
    latest_game_view = extract_game_view_bgr(selected[-1]) if selected else None
    text = {
        "current_step": int(state.counters.get("current_step", 0)),
        "operating_mode": "visual_only_no_ram",
        "prompt_image_summary": {
            "full_capture_frames": len(selected),
            "full_capture_detail": detail,
            "jpeg_quality": config.jpeg_quality,
            "latest_game_view_zoom_included": bool(config.include_latest_game_view and latest_game_view is not None),
            "latest_game_view_zoom_detail": "high" if config.include_latest_game_view and latest_game_view is not None else None,
        },
        "early_game_bootstrap_guide": EARLY_GAME_BOOTSTRAP_GUIDE,
        "visual_change_summary": visual_change_summary,
        "stagnation_summary": build_stagnation_summary(state.history, visual_change_summary),
        "navigation_summary": build_navigation_summary(state.history),
        "memory": selected_memory,
        "memory_selection": {
            "entries_sent": len(selected_memory),
            "entries_available": len(state.memory),
            "policy": "current-objective and recent-history relevance; newer local route memories win over older contradictory probes",
        },
        "objectives": state.objectives,
        "maps": state.maps,
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

    for idx, frame in enumerate(selected, start=1):
        content.append({"type": "input_text", "text": f"frame_{idx}: full capture, chronological order"})
        content.append(
            {
                "type": "input_image",
                "image_url": to_data_url_bgr(frame, config.jpeg_quality),
                "detail": detail,
            }
        )

    if config.include_latest_game_view and selected:
        latest = selected[-1]
        game_view = latest_game_view
        if game_view is not None:
            game_view = upscale_for_prompt(game_view, config.game_view_min_width)
            content.append(
                {
                    "type": "input_text",
                    "text": "latest_game_view_zoom: auto-detected active game viewport from the newest frame; use this for tile-level terrain such as ledges, openings, stairs, and doors.",
                }
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": to_data_url_bgr(game_view, config.jpeg_quality),
                    "detail": "high",
                }
            )
    return content


def normalize_key(key: Any) -> str:
    normalized = str(key).strip().upper()
    if normalized == "A_UNTIL_END_OF_DIALOG":
        return normalized
    return normalized if normalized in ALLOWED_KEYS else "WAIT"


def normalize_scene_type(scene_type: Any) -> str:
    normalized = str(scene_type).strip().lower()
    return normalized if normalized in SCENE_TYPES else "unclear"


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
            for start in range(0, len(keys), MAX_KEYS_PER_KEY_PRESS):
                chunk = keys[start : start + MAX_KEYS_PER_KEY_PRESS]
                if chunk:
                    normalized_actions.append({"type": "key_press", "keys": chunk})
        elif action_type in {"write_memory", "delete_memory", "update_objectives", "add_marker", "delete_marker"}:
            normalized_actions.append(action)

    if not any(a.get("type") == "key_press" for a in normalized_actions):
        normalized_actions.append({"type": "key_press", "keys": ["WAIT"]})

    return {
        "scene_type": normalize_scene_type(data.get("scene_type", "unclear")),
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
                "others": action.get("others", []),
            }
        elif action_type == "add_marker":
            marker = normalize_marker(action, int(state.counters.get("current_step", 0)))
            if marker is not None:
                upsert_marker(state, marker)
        elif action_type == "delete_marker":
            delete_marker(state, action)


def key_sequence_from_actions(actions: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    for action in actions:
        if action.get("type") != "key_press":
            continue
        for key in action.get("keys", []):
            keys.append(normalize_key(key))
    return keys or ["WAIT"]


def _entry_keys(entry: dict[str, Any]) -> list[str]:
    raw_keys = entry.get("keys", [])
    if not isinstance(raw_keys, list):
        return []
    return [normalize_key(key) for key in raw_keys]


def _has_recent_a_like_without_movement(history: list[dict[str, Any]], lookback: int = 4) -> bool:
    saw_a_like = False
    for entry in reversed(history[-lookback:]):
        keys = _entry_keys(entry)
        if any(key in MOVEMENT_KEYS for key in keys):
            return False
        if any(key in A_LIKE_KEYS for key in keys):
            saw_a_like = True
    return saw_a_like


def _decision_text(decision: dict[str, Any]) -> str:
    return (
        str(decision.get("chat_message", ""))
        + " "
        + str(decision.get("step_details", ""))
    ).lower()


def _decision_mentions_object_interaction(decision: dict[str, Any]) -> bool:
    text = _decision_text(decision)
    return any(term in text for term in OBJECT_INTERACTION_TERMS)


def _decision_mentions_story_dialog(decision: dict[str, Any]) -> bool:
    text = _decision_text(decision)
    return any(term in text for term in STORY_DIALOG_TERMS)


def _is_repeated_static_object_dialog(
    state: HarnessState,
    decision: dict[str, Any],
    visual_summary: dict[str, Any],
) -> bool:
    if normalize_scene_type(decision.get("scene_type", "unclear")) != "dialog":
        return False
    if not _decision_mentions_object_interaction(decision):
        return False
    if _decision_mentions_story_dialog(decision):
        return False

    stagnation = build_stagnation_summary(state.history, visual_summary)
    return bool(stagnation["low_change_now"] and stagnation["recent_a_like_count"] >= 2)


def _decision_has_strong_a_context(decision: dict[str, Any]) -> bool:
    scene_type = normalize_scene_type(decision.get("scene_type", "unclear"))
    if scene_type == "overworld":
        return False
    if _decision_mentions_object_interaction(decision) and not _decision_mentions_story_dialog(decision):
        return False
    if scene_type in {"dialog", "menu", "battle"}:
        return True

    text = _decision_text(decision)

    negative_context = (
        "no dialog",
        "no text",
        "no menu",
        "nothing to advance",
        "overworld",
        "facing an npc",
        "facing a person",
        "facing an object",
        "talk to",
        "interact with",
    )
    if any(phrase in text for phrase in negative_context):
        return False

    positive_context = (
        "clear dialog",
        "dialogue box",
        "text box",
        "battle text",
        "continuation arrow",
        "confirmation",
        "confirm",
        "choice",
        "prompt",
        "menu",
        "keyboard",
        "naming",
        "yes/no",
    )
    return any(phrase in text for phrase in positive_context)


def _decision_has_required_overworld_interaction(decision: dict[str, Any]) -> bool:
    text = _decision_text(decision)

    blocked_context = (
        "bedroom",
        "furniture",
        "pc",
        "tv",
        "snes",
        "sign",
        "decoration",
        "same npc",
        "same object",
        "talk again",
    )
    if any(phrase in text for phrase in blocked_context):
        return False

    required_context = (
        "required interaction",
        "story interaction",
        "starter",
        "pokeball",
        "professor oak",
        "oak",
        "confirm the highlighted",
        "confirm the selected",
        "interact with the door",
        "use the stairs",
    )
    return any(phrase in text for phrase in required_context)


def _decision_has_adjacent_facing_context(decision: dict[str, Any]) -> bool:
    text = _decision_text(decision)

    negative_context = (
        "not adjacent",
        "not next to",
        "not facing",
        "wrong facing",
        "wrong direction",
        "too far",
        "far away",
        "gap",
        "one or more tiles away",
        "more than one tile",
        "diagonal",
        "approach",
        "move toward",
        "move closer",
        "need to face",
        "turn toward",
    )
    if any(phrase in text for phrase in negative_context):
        return False

    adjacent_context = (
        "directly adjacent",
        "adjacent to",
        "adjacent above",
        "adjacent below",
        "adjacent left",
        "adjacent right",
        "next to",
        "beside",
        "directly beside",
        "neighboring tile",
        "in front of",
        "directly in front",
        "directly below",
        "directly above",
        "directly left",
        "directly right",
        "clean vertical alignment",
        "clean horizontal alignment",
        "vertical alignment",
        "horizontal alignment",
        "interaction range",
    )
    facing_context = (
        "facing the target",
        "facing it",
        "facing him",
        "facing her",
        "facing the npc",
        "facing the object",
        "facing the door",
        "facing the pokeball",
        "facing upward",
        "facing up",
        "facing downward",
        "facing down",
        "facing left",
        "facing right",
        "correctly facing",
        "already facing",
        "aligned to interact",
        "interaction setup",
        "correct setup for an interaction",
        "correct interaction setup",
    )
    return any(phrase in text for phrase in adjacent_context) and any(
        phrase in text for phrase in facing_context
    )


def _decision_mentions_blocked_upward_ledge(decision: dict[str, Any]) -> bool:
    text = _decision_text(decision)
    has_ledge = any(term in text for term in LEDGE_TERMS)
    has_upward_block = any(term in text for term in UPWARD_BLOCK_TERMS)
    return has_ledge and has_upward_block


def _decision_mentions_ledge_opening_probe(decision: dict[str, Any]) -> bool:
    text = _decision_text(decision)
    return any(term in text for term in LEDGE_TERMS) and any(
        term in text for term in LEDGE_OPENING_PROBE_TERMS
    )


def _recent_lateral_movement(history: list[dict[str, Any]], lookback: int = 3) -> bool:
    for entry in reversed(history[-lookback:]):
        keys = _entry_keys(entry)
        if any(key in {"LEFT", "RIGHT"} for key in keys):
            return True
    return False


def _decision_mentions_lateral_ledge_progress(decision: dict[str, Any]) -> bool:
    text = _decision_text(decision)
    has_ledge = any(term in text for term in LEDGE_TERMS)
    has_progress = any(term in text for term in LATERAL_LEDGE_PROGRESS_TERMS)
    has_stop = any(term in text for term in LATERAL_LEDGE_STOP_TERMS)
    return has_ledge and has_progress and not has_stop


def _decision_mentions_doorway_loop_risk(decision: dict[str, Any]) -> bool:
    text = _decision_text(decision)
    negative_context = (
        "no doorway",
        "no door",
        "no entrance",
        "not near a doorway",
        "not near the doorway",
        "away from the doorway",
        "clear of the doorway",
        "clear of any doorway",
    )
    if any(term in text for term in negative_context):
        return False
    return any(term in text for term in DOORWAY_LOOP_TERMS) and (
        "re-trigger" in text
        or "retrigger" in text
        or "doorway" in text
        or "entrance" in text
        or "frontage" in text
    )


def _recent_lateral_only_count(history: list[dict[str, Any]], direction: str, lookback: int = 5) -> int:
    count = 0
    for entry in reversed(history[-lookback:]):
        keys = [key for key in _entry_keys(entry) if key != "WAIT"]
        if keys and all(key == direction for key in keys):
            count += 1
            continue
        break
    return count


def _opposite_lateral_direction(direction: str) -> str:
    return "LEFT" if direction == "RIGHT" else "RIGHT"


def _lateral_commitment(history: list[dict[str, Any]], lookback: int = 5) -> dict[str, Any]:
    direction = ""
    presses = 0
    decisions = 0

    for entry in reversed(history[-lookback:]):
        keys = [key for key in _entry_keys(entry) if key != "WAIT"]
        lateral = [key for key in keys if key in {"LEFT", "RIGHT"}]
        if not lateral:
            if keys:
                break
            continue
        if len(lateral) != len(keys) or len(set(lateral)) != 1:
            break
        entry_direction = lateral[0]
        if not direction:
            direction = entry_direction
        if entry_direction != direction:
            break
        presses += len(lateral)
        decisions += 1

    return {
        "direction": direction,
        "presses": presses,
        "decisions": decisions,
        "is_short": bool(direction) and presses < 5 and decisions < 3,
    }


def _decision_mentions_hard_lateral_stop(decision: dict[str, Any]) -> bool:
    text = _decision_text(decision)
    return any(term in text for term in LATERAL_LEDGE_STOP_TERMS)


def _remove_upward_ledge_attempt(
    history: list[dict[str, Any]],
    keys: list[str],
) -> list[str]:
    before_up: list[str] = []
    for key in keys:
        if key == "UP":
            break
        if key != "WAIT":
            before_up.append(key)

    if before_up:
        return before_up

    recent_moves = [
        key
        for entry in reversed(history[-6:])
        for key in _entry_keys(entry)
        if key in MOVEMENT_KEYS
    ]
    for candidate in ("RIGHT", "LEFT", "DOWN"):
        if candidate not in recent_moves[:3]:
            return [candidate]
    return ["RIGHT"]


def _has_lateral_setup_before_up(keys: list[str]) -> bool:
    for key in keys:
        if key == "UP":
            return False
        if key in {"LEFT", "RIGHT", "DOWN"}:
            return True
    return False


def guard_against_one_way_ledge(
    state: HarnessState,
    decision: dict[str, Any],
    keys: list[str],
) -> tuple[list[str], str | None]:
    normalized_keys = [normalize_key(key) for key in keys]
    if normalize_scene_type(decision.get("scene_type", "unclear")) != "overworld":
        return normalized_keys, None
    if "UP" not in normalized_keys:
        return normalized_keys, None
    if not _decision_mentions_blocked_upward_ledge(decision):
        return normalized_keys, None
    if _has_lateral_setup_before_up(normalized_keys):
        return normalized_keys, None
    if _decision_mentions_ledge_opening_probe(decision) and _recent_lateral_movement(state.history):
        return normalized_keys, None

    replacement = _remove_upward_ledge_attempt(state.history, normalized_keys)
    note = (
        "one-way ledge guard removed UP because brown ledges cannot be climbed "
        "from below; route laterally to an opening instead"
    )
    return replacement, note


def guard_against_lateral_ledge_loop(
    state: HarnessState,
    decision: dict[str, Any],
    keys: list[str],
) -> tuple[list[str], str | None]:
    normalized_keys = [normalize_key(key) for key in keys]
    if normalize_scene_type(decision.get("scene_type", "unclear")) != "overworld":
        return normalized_keys, None
    if "UP" in normalized_keys:
        return normalized_keys, None
    lateral_keys = [key for key in normalized_keys if key in {"LEFT", "RIGHT"}]
    non_wait_keys = [key for key in normalized_keys if key != "WAIT"]
    if not lateral_keys or len(lateral_keys) != len(non_wait_keys):
        return normalized_keys, None
    if len(set(lateral_keys)) != 1:
        return normalized_keys, None
    if not _decision_mentions_lateral_ledge_progress(decision):
        return normalized_keys, None

    direction = lateral_keys[0]
    if len(lateral_keys) < 3 and _recent_lateral_only_count(state.history, direction) < 2:
        return normalized_keys, None

    note = (
        "ledge opening guard appended UP because repeated lateral movement along "
        "an open ledge lane should test the vertical opening instead of continuing sideways"
    )
    return normalized_keys + ["UP"], note


def guard_against_premature_lateral_reversal(
    state: HarnessState,
    decision: dict[str, Any],
    keys: list[str],
) -> tuple[list[str], str | None]:
    normalized_keys = [normalize_key(key) for key in keys]
    if normalize_scene_type(decision.get("scene_type", "unclear")) != "overworld":
        return normalized_keys, None

    non_wait_keys = [key for key in normalized_keys if key != "WAIT"]
    lateral_keys = [key for key in non_wait_keys if key in {"LEFT", "RIGHT"}]
    if not lateral_keys or len(lateral_keys) != len(non_wait_keys):
        return normalized_keys, None
    if len(set(lateral_keys)) != 1:
        return normalized_keys, None

    current_direction = lateral_keys[0]
    commitment = _lateral_commitment(state.history)
    previous_direction = str(commitment.get("direction", ""))
    if not previous_direction:
        return normalized_keys, None
    if current_direction != _opposite_lateral_direction(previous_direction):
        return normalized_keys, None
    if not commitment.get("is_short"):
        return normalized_keys, None
    if _decision_mentions_hard_lateral_stop(decision):
        return normalized_keys, None

    replacement_length = max(len(lateral_keys), 3)
    replacement = [previous_direction] * replacement_length
    note = (
        "lateral exploration guard prevented an early reversal; continue the same "
        "side search for a meaningful distance before giving up unless a hard wall or route edge is visible"
    )
    return replacement, note


def guard_against_doorway_route_loop(
    state: HarnessState,
    decision: dict[str, Any],
    keys: list[str],
) -> tuple[list[str], str | None]:
    normalized_keys = [normalize_key(key) for key in keys]
    if normalize_scene_type(decision.get("scene_type", "unclear")) != "overworld":
        return normalized_keys, None
    if "UP" not in normalized_keys:
        return normalized_keys, None

    navigation = build_navigation_summary(state.history)
    if not navigation["doorway_loop_risk"] and not _decision_mentions_doorway_loop_risk(decision):
        return normalized_keys, None

    non_wait = [key for key in normalized_keys if key != "WAIT"]
    if non_wait and non_wait[0] in {"LEFT", "RIGHT"}:
        return normalized_keys, None

    recent_moves = [
        key
        for entry in reversed(state.history[-8:])
        for key in _entry_keys(entry)
        if key in MOVEMENT_KEYS
    ]
    side = "LEFT" if recent_moves.count("LEFT") <= recent_moves.count("RIGHT") else "RIGHT"
    replacement = [side, side, "UP"]
    note = (
        "doorway route guard replaced UP-first movement because recent history suggests "
        "a building entrance loop; move sideways out of the doorway column before routing north"
    )
    return replacement, note


def _truncate_or_replace_unsafe_overworld_a(
    history: list[dict[str, Any]],
    keys: list[str],
    fallback_move: str | None,
) -> list[str]:
    before_a: list[str] = []
    for key in keys:
        if key in A_LIKE_KEYS:
            break
        if key != "WAIT":
            before_a.append(key)

    if before_a:
        return before_a

    move = normalize_key(fallback_move or fallback_movement_for_recent_history(history))
    if move not in MOVEMENT_KEYS:
        move = "DOWN"
    return [move]


def fallback_movement_for_recent_history(history: list[dict[str, Any]]) -> str:
    recent_moves: list[str] = []
    for entry in reversed(history[-6:]):
        recent_moves.extend(key for key in _entry_keys(entry) if key in MOVEMENT_KEYS)

    if not recent_moves:
        return "DOWN"

    for candidate in ("DOWN", "RIGHT", "UP", "LEFT"):
        if candidate not in recent_moves[:4]:
            return candidate

    alternatives = {
        "UP": "RIGHT",
        "RIGHT": "DOWN",
        "DOWN": "LEFT",
        "LEFT": "UP",
    }
    return alternatives.get(recent_moves[0], "RIGHT")


def alternate_key_for_stagnation(history: list[dict[str, Any]], keys: list[str]) -> str | None:
    recent_keys = [key for entry in history[-5:] for key in _entry_keys(entry) if key != "WAIT"]
    current_non_wait = [key for key in keys if key != "WAIT"]
    if not recent_keys or not current_non_wait:
        return None

    current_first = current_non_wait[0]
    if len(recent_keys) >= 2 and recent_keys[-1] == current_first and recent_keys[-2] == current_first:
        if current_first in A_LIKE_KEYS:
            return fallback_movement_for_recent_history(history)
        if current_first in MOVEMENT_KEYS:
            alternatives = {
                "UP": "RIGHT",
                "RIGHT": "DOWN",
                "DOWN": "LEFT",
                "LEFT": "UP",
            }
            return alternatives[current_first]

    return None


def guard_against_stagnation(
    state: HarnessState,
    decision: dict[str, Any],
    keys: list[str],
    visual_summary: dict[str, Any],
) -> tuple[list[str], str | None]:
    normalized_keys = [normalize_key(key) for key in keys]
    if _is_repeated_static_object_dialog(state, decision, visual_summary):
        replacement = fallback_movement_for_recent_history(state.history)
        note = (
            "stagnation guard changed repeated object-dialog advancement to movement "
            "because recent A-like inputs did not produce visual progress"
        )
        return [normalize_key(replacement)], note

    if _decision_has_strong_a_context(decision):
        return normalized_keys, None

    stagnation = build_stagnation_summary(state.history, visual_summary)
    if not stagnation["is_stagnant"]:
        return normalized_keys, None

    replacement = alternate_key_for_stagnation(state.history, normalized_keys)
    if replacement is None and any(key in A_LIKE_KEYS for key in normalized_keys):
        replacement = fallback_movement_for_recent_history(state.history)

    if replacement is None:
        return normalized_keys, None

    note = (
        "stagnation guard changed the action because recent history and visual "
        f"diffs indicate little progress: {stagnation}"
    )
    return [normalize_key(replacement)], note


def guard_against_reinteraction_loop(
    state: HarnessState,
    decision: dict[str, Any],
    keys: list[str],
    fallback_move: str | None = None,
) -> tuple[list[str], str | None]:
    normalized_keys = [normalize_key(key) for key in keys]
    first_a_index = next(
        (idx for idx, key in enumerate(normalized_keys) if key in A_LIKE_KEYS),
        None,
    )
    if first_a_index is None:
        return normalized_keys, None

    if _decision_has_strong_a_context(decision):
        return normalized_keys, None

    if normalize_scene_type(decision.get("scene_type", "unclear")) == "overworld":
        has_required_context = _decision_has_required_overworld_interaction(decision)
        has_adjacent_facing_context = _decision_has_adjacent_facing_context(decision)
        if (
            _has_recent_a_like_without_movement(state.history)
            or not has_required_context
            or not has_adjacent_facing_context
        ):
            replacement = _truncate_or_replace_unsafe_overworld_a(
                state.history,
                normalized_keys,
                fallback_move,
            )
            note = (
                "anti-loop guard removed an overworld A-like input because interaction "
                "requires a clearly adjacent target and correct facing"
            )
            return replacement, note

    first_action_key = next((key for key in normalized_keys if key != "WAIT"), "WAIT")
    if first_action_key in A_LIKE_KEYS and (
        normalize_scene_type(decision.get("scene_type", "unclear")) == "overworld"
        and not _decision_has_required_overworld_interaction(decision)
    ):
        move = fallback_move or fallback_movement_for_recent_history(state.history)
        note = (
            "anti-loop guard replaced an A-like input with movement because the model "
            "classified the screen as overworld without a clear required interaction"
        )
        return [normalize_key(move)], note

    if not _has_recent_a_like_without_movement(state.history):
        return normalized_keys, None

    if _decision_has_strong_a_context(decision):
        return normalized_keys, None

    move = normalize_key(fallback_move or fallback_movement_for_recent_history(state.history))
    if move not in MOVEMENT_KEYS:
        move = "DOWN"
    note = (
        "anti-loop guard replaced an A-like input with movement because recent history "
        "already used A/A_UNTIL_END_OF_DIALOG and the new decision did not cite a clear "
        "dialog, prompt, menu, or confirmation context"
    )
    return [move], note
