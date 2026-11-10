"""Evidence-based replay: recorded RGB, execution phases and actual tool receipts.

Uses the previous gpt_as_policy RoboCasa recorder's step-aligned three-camera
convention. This renderer never runs a model or changes a saved episode.
"""

import hashlib
import json
import math
import subprocess
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data import read_events

FPS = 20
SIZE = (1280, 800)
COLORS = {"VLA": "#4fd1c5", "Harness": "#6cb6ff", "Decision": "#f6c760",
          "Astra": "#c6a0ff", "Code Policy": "#f69873", "Verification": "#9cdb8c",
          "Runtime": "#aab8cc", "Flora": "#f0abfc"}


def compact(value, limit=250):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "…"


def build_timeline(rows):
    """Order by recorded events, not observation step (many calls share a step)."""
    if not rows:
        raise ValueError("No recorded events")
    start = rows[0]["time_ns"]
    obs = None
    source, instruction, subtask = "none", "", "task"
    decision, owner = "not run", "none"
    frames, cards = 0, []
    for row in rows:
        kind = row["kind"]
        component = tool = phase = description = None
        seconds = 0
        if kind == "observation":
            previous = obs
            obs = row
            component, tool = "Harness", "Harness.execute -> Environment.step"
            if previous is None:
                tool = "Environment.reset / observe"
            phase = "Physical execution" if row["step"] else "Initial observation"
            description = instruction or row.get("facts", {}).get("task_instruction", "")
            seconds = max(1 / FPS, row["sim_time"] - previous["sim_time"]) if previous else 1.0
            if row.get("success"):
                phase, description = "Task success observed", "Native environment success signal is true."
        elif kind == "task_initialized":
            instruction = row["task"]["instruction"]
            subtask = row["task"]["subtasks"][0]["name"]
            component, tool, phase = "Runtime", "task_initialized", "Task setup"
            description, seconds = instruction, 0.8
        elif kind == "control_claim":
            owner = row["owner"]
            component, tool, phase = "Harness", "claim", "Control handoff"
            description = f"Owner: {owner}; discarded queued actions: {row['discarded_steps']}. No motion implied."
            seconds = 0.3
        elif kind == "action_command":
            source, instruction = row["source"], row["instruction"]
        elif kind == "vla_inference_start":
            instruction = row["instruction"]
            component, tool, phase = "VLA", "Pi05.infer (WebSocket)", "Action generation"
            description, seconds = instruction, 0.15
        elif kind == "vla_replay":
            component, tool, phase = "VLA", "Recorded VLA commands", "Failure reproduction"
            description, seconds = (f"Replay of a saved command prefix in live physics; "
                                    f"measured state error={row['state_error']:.3g}. No new VLA inference.", 0.15)
        elif kind == "vla_inference_end" and row.get("error"):
            component, tool, phase = "VLA", "Pi05.infer", "Inference failed"
            description, seconds = row["error"], 1.2
        elif kind == "decision":
            value = row["decision"]
            decision = value["label"]
            probabilities = value.get("probabilities", {})
            raw_choice = max(probabilities, key=probabilities.get) if probabilities else "unavailable"
            component, tool, phase = "Decision", "Decision3.decide", "Execution review"
            description = (f"Raw choice {raw_choice} -> route {decision}; evaluated step {value['binding']['observation_step']}; "
                           f"{row['seconds']:.2f}s; {value.get('fallback_reason') or 'model classification'}")
            if row.get('backend') == 'flora':
                component, tool = 'Flora', 'Outcome score routing'
                description = (f"Score {row.get('score')} -> review route {decision}; "
                               f"snapshot step {value['binding']['observation_step']}; "
                               f"{row['seconds']:.2f}s; {value.get('fallback_reason')}. No motion authorized.")
            seconds = 0.7
        elif kind == "codex_turn_start":
            component, tool = "Astra", "Codex turn/start"
            intent = row.get("context", {}).get("intent")
            phase = "Initial planning" if intent == "initialize_plan" else (
                "Read-only review" if intent == "review_running" else "Intervention reasoning")
            description = ("VLA retains control while Astra reviews a frozen snapshot." if intent == "review_running"
                           else "Robot execution is paused while Astra reviews evidence.")
            seconds = 1.2
        elif kind in {"review_begin", "review_end", "review_discarded", "review_failed", "review_cancelled"}:
            component, tool, phase = "Astra", kind, "Read-only supervision"
            description, seconds = compact(row), 0.7
        elif kind == "review_refresh":
            component, tool, phase = "Runtime", kind, "Fresh read-only review"
            description, seconds = "Expired concern triggers a new snapshot review; VLA retains control.", 0.7
        elif kind in {"attempt_interrupted", "attempt_resumed"}:
            component, tool, phase = "Runtime", kind, "Attempt state update"
            description, seconds = compact(row), 0.7
        elif kind in {"flora_request", "flora_result", "flora_cache_hit", "flora_unavailable", "flora_cancelled"}:
            component, tool, phase = "Flora", "evaluate_subtask", "Subtask outcome assessment"
            assessment = row.get("result", row)
            binding = assessment.get("binding", {})
            description = (f"{assessment.get('status', kind)}; score={assessment.get('score', 'unknown')}; "
                           f"assessed snapshot step={binding.get('observation_step', 'unknown')}. "
                           "Historical model assessment; does not transfer control or verify completion.")
            seconds = 0.7
        elif kind == "codex_tool_start":
            component, tool, phase = "Astra", row["name"], "Tool request"
            description, seconds = "Host validates arguments and dispatches the registered tool.", 0.35
        elif kind == "codex_tool":
            component, tool, phase = "Astra", row["name"], "Tool result"
            description, seconds = compact(row["result"]), 0.7
        elif kind == "intervention_plan":
            p = row["plan"]
            component, tool, phase = "Astra", "submit_intervention / " + p["kind"], "Accepted intervention"
            description, seconds = p["reason"], 1.2
            if p.get("remaining_plan"):
                subtask = p["remaining_plan"][0]["name"]
        elif kind == "supervision_rule":
            component, tool, phase = "Runtime", kind, "Numeric evidence for review"
            description = compact(row.get("signals", []))
        elif kind == 'subtask_transition':
            component, tool, phase = 'Runtime', kind, 'Verified executable subtask transition'
            next_task = row.get('next_subtask')
            if next_task:
                subtask = next_task['name']
                instruction = next_task['instruction']
            description = compact(dict(completed=row['completed'], next_subtask=next_task,
                                       next_executor=row.get('next_executor')))
            seconds = 1.2
        elif kind in {'subtask_entry', 'subtask_boundary_ready'}:
            component, tool, phase = 'Harness', kind, 'Registered subtask verification'
            description, seconds = compact(row), 0.7
        elif kind in {"phase_plan", "phase_assessment", "grounding", "supervision_audit"}:
            component, tool, phase = "Astra", kind, "Evidence and phase assessment"
            description = compact({k: v for k, v in row.items() if k not in {"id", "time_ns"}})
        elif kind in {"executor_assignment", "code_subtask_complete", "code_subtask_abandoned", "recovery_session_budget"}:
            component, tool, phase = "Runtime", kind, "Subtask executor assignment"
            description = compact({k: v for k, v in row.items() if k not in {"id", "time_ns"}})
        elif kind in {"stage_begin", "stage_end"}:
            stage = row["stage"]
            component, tool, phase = "Code Policy", kind, stage["name"]
            description = f"{stage['status']}; steps {stage.get('start_step', 'unknown')} -> {stage.get('end_step')}; " + (
                stage.get("error") or "Program stage; completion of this stage alone does not verify the subtask.")
            seconds = 0.7
        elif kind == "code_version":
            component, tool, phase = "Code Policy", "run program", "Execute " + row["program_version"]
            description, seconds = "Completion condition: " + row["completion_condition"], 0.7
        elif kind == "recovery_result":
            component, tool, phase = "Code Policy", kind, "Program outcome"
            description = (f"{row['program_version']}: {row['result']}; verified={row['verified']}; "
                           f"{row['executed_steps']} executed steps; "
                           f"failed stage={row.get('failed_stage')}; error={row.get('error')}")
            seconds = 1.0
        elif kind == "recovery_note":
            component, tool, phase = "Code Policy", "note", "Program-reported observations"
            description, seconds = compact(row.get("facts", [])), 0.7
        elif kind == "recovery_stage":
            component, tool, phase = "Code Policy", kind, "Program recovery"
            description, seconds = compact(row), 0.7
        elif kind == "recovery_tool_request":
            component, tool, phase = "Code Policy", row["operation"], "Program tool request"
            description, seconds = compact(row), 0.25
        elif kind == "recovery_tool_result":
            component, tool, phase = "Code Policy", row["operation"], "Program tool result"
            description, seconds = compact(row["result"]), 0.35
        elif kind == "verification":
            component, tool, phase = "Verification", "Harness.verify", "Completion verification"
            description, seconds = compact(row["result"]), 0.8
        elif kind == "runtime_error":
            component, tool, phase = "Runtime", "runtime_error", "Run interrupted"
            description, seconds = row["error"], 1.5
        elif kind == "stop":
            component, tool, phase = "Harness", "stop", "Execution stopped"
            description, seconds = row["reason"], 1.0
        elif kind == "episode_end":
            component, tool, phase = "Runtime", "episode_end", "Episode result"
            description, seconds = compact(row.get("result", row)), 2.0
        if component and obs is not None:
            duration_frames = max(1, round(seconds * FPS))
            cards.append(dict(event_id=row["id"], kind=kind, component=component, tool=tool,
                              phase=phase, description=description, step=obs["step"],
                              sim_seconds=obs["sim_time"], wall_seconds=(row["time_ns"] - start) / 1e9,
                              owner=owner, action_source=source, instruction=instruction, subtask=subtask,
                              decision=decision, observation=obs["path"],
                              video_start=frames / FPS, video_end=(frames + duration_frames) / FPS,
                              duration_frames=duration_frames, details=row))
            frames += duration_frames
    return cards


def overview_timeline(cards):
    """Keep every physical frame and stage; compress repetitive tool detail for review."""
    selected, seen, frames = [], set(), 0
    high_level = {"write_program", "patch_program", "patch_program_batch", "edit_program", "check_program", "search_skills", "evaluate_subtask",
                  "submit_intervention", "submit_review", "write_skill_package", "patch_skill_package",
                  "import_skill_program", "adapt_program_goal", "check_skill", "set_phase_plan", "assess_phase",
                  "ground_target", "measure_targets", "fit_geometry", "read_recovery_trace"}
    for card in cards:
        kind, row = card["kind"], card["details"]
        if kind in {"stage_begin", "code_version"}:
            seen.clear()
        if kind in {"codex_tool", "codex_tool_start"} and row["name"] not in high_level:
            continue
        if kind == "recovery_tool_result":
            if not isinstance(row.get("result"), dict) or not row["result"].get("error"):
                continue
        if kind == "recovery_tool_request":
            op = row["operation"]
            if op in seen or op in {"stage", "stage_end", "result", "note"}:
                continue
            seen.add(op)
        if kind == "verification" and row.get("result", {}).get("passed") is not True:
            continue
        item = dict(card, video_start=frames / FPS,
                    video_end=(frames + card["duration_frames"]) / FPS)
        selected.append(item)
        frames += card["duration_frames"]
    return selected


def episode_summary(root, rows):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    result = json.loads((root / "result.json").read_text())
    config = manifest["config"]
    first = next(r for r in rows if r["kind"] == "observation")
    with np.load(root / first["path"], allow_pickle=False) as arr:
        fingerprint = {key: hashlib.sha256(np.ascontiguousarray(arr[key]).tobytes()).hexdigest()
                       for key in sorted(arr.files)}
    times = [r["seconds"] for r in rows if r["kind"] == "vla_inference_end" and not r.get("error")]
    observations = [r for r in rows if r["kind"] == "observation"]
    commands = [r for r in rows if r["kind"] == "action_command"]
    receipts = [r for r in rows if r["kind"] == "action_executed"]
    native = observations[-1]
    audit = dict(
        contiguous_observation_steps=[r["step"] for r in observations] == list(range(result["control_steps"] + 1)),
        all_commands_acknowledged={r["id"] for r in commands} == {r["command_id"] for r in receipts},
        no_duplicate_receipts=len({r["command_id"] for r in receipts}) == len(receipts),
        result_matches_native_success=result["success"] == native["success"],
        executed_steps_match_receipts=len(receipts) == result["control_steps"],
        configured_chunk_steps=config["runtime"]["chunk_steps"],
        native_horizon=native.get("facts", {}).get("native_horizon"),
    )
    mode = config["runtime"].get("experiment_label", "default")
    if mode == "default":
        mode = "supervised" if config["runtime"]["supervision_enabled"] else "baseline"
    return dict(task=config["environment"]["task_name"], seed=config["environment"]["seed"], mode=mode,
                result=result, initial_fingerprint=fingerprint, vla_calls=len(times),
                vla_seconds=sum(times), tools=[r["name"] for r in rows if r["kind"] == "codex_tool"],
                instruction=next((r["task"]["instruction"] for r in rows if r["kind"] == "task_initialized"), ""),
                source=str(root.resolve()), checkpoint=config["vla"]["checkpoint"], audit=audit)


def canvas(root, card, summary):
    frame = Image.new("RGB", SIZE, "#101724")
    draw = ImageDraw.Draw(frame)
    font_file = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    fonts = {n: ImageFont.truetype(font_file, n) for n in (14, 17, 20, 28)}
    color = COLORS[card["component"]]
    draw.rectangle((0, 0, 1280, 7), fill=color)
    draw.text((24, 22), summary["task"], font=fonts[28], fill="#f2f5fb")
    draw.text((24, 65), f"{summary['mode'].upper()}  |  seed {summary['seed']}  |  Recorded evidence replay",
              font=fonts[17], fill="#aebed3")
    with np.load(Path(root) / card["observation"], allow_pickle=False) as arrays:
        for x, camera in zip((24, 440, 856), ("left", "right", "wrist")):
            img = Image.fromarray(arrays["rgb/" + camera]).resize((400, 400), Image.Resampling.BILINEAR)
            frame.paste(img, (x, 126))
            draw.text((x, 100), camera.upper(), font=fonts[14], fill="#aebed3")
    draw.rounded_rectangle((24, 543, 1256, 750), radius=12, fill="#192436", outline="#2d4058")
    draw.text((42, 558), card["phase"], font=fonts[28], fill=color)
    draw.text((42, 600), "Tool / interface: " + card["tool"], font=fonts[20], fill="#f2f5fb")
    description = card["description"].replace("\n", " ")
    for i, line in enumerate(textwrap.wrap(description, 125)[:3]):
        draw.text((42, 636 + i * 25), line, font=fonts[17], fill="#bdcce0")
    footer = (f"step {card['step']}  |  simulation {card['sim_seconds']:.2f}s  |  wall {card['wall_seconds']:.1f}s"
              f"  |  last action: {card['action_source']}  |  Decision: {card['decision']}  |  event #{card['event_id']}")
    draw.text((24, 771), footer, font=fonts[14], fill="#9dafc7")
    return frame


def render_episode(root, output, *, overview=False, label=None):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = read_events(root)
    cards, summary = build_timeline(rows), episode_summary(root, rows)
    if label:
        summary["configured_mode"] = summary["mode"]
        summary["mode"] = label
    original_card_count = len(cards)
    if overview:
        cards = overview_timeline(cards)
    if not cards:
        raise ValueError("No renderable observations")
    metadata = dict(schema="vla_agent.replay.v1", summary=summary, timeline=cards,
                    detail="overview" if overview else "full", omitted_tool_cards=original_card_count-len(cards),
                    fps=FPS, timing="Physical frames retain 20 Hz. Reasoning/tool waits are compressed; wall time remains visible."
                    + (" Overview omits repetitive tool detail; source logs retain every call." if overview else ""),
                    phase_semantics="Execution phases from logs, not inferred grasp/subtask success labels.",
                    source_events_sha256=hashlib.sha256((root / "events.jsonl").read_bytes()).hexdigest())
    (output / "timeline.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    canvas(root, cards[0], summary).save(output / "poster.jpg", quality=92)
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{SIZE[0]}x{SIZE[1]}", "-r", str(FPS), "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-threads", "2",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output / "replay.mp4")]
    with (output / "ffmpeg.log").open("wb") as err:
        process = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=err)
        try:
            for card in cards:
                raw = canvas(root, card, summary).tobytes()
                for _ in range(card["duration_frames"]):
                    process.stdin.write(raw)
            process.stdin.close()
            if process.wait(timeout=120):
                raise RuntimeError("Video encoding failed; inspect ffmpeg.log")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
        "stream=width,height,nb_read_frames,duration", "-of", "json", str(output / "replay.mp4"),
    ], text=True))["streams"][0]
    assert int(probe["nb_read_frames"]) == sum(c["duration_frames"] for c in cards)
    assert math.isclose(float(probe["duration"]), cards[-1]["video_end"], abs_tol=0.06)
    (output / "video-check.json").write_text(json.dumps(probe, indent=2))
    write_page(output, metadata)
    return summary


def write_page(output, metadata):
    html = Path(__file__).with_name("replay.html").read_text()
    payload = json.dumps(metadata, ensure_ascii=False).replace("<", "\\u003c")
    (Path(output) / "index.html").write_text(html.replace("__REPORT_JSON__", payload))
