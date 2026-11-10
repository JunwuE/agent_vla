"""Audit completed episodes without running models or changing runtime state.

Turn durations include model, provider, network and host tool time. Background
reviews and Flora overlap execution; they must not be added to episode wall time.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def stats(values):
    values = list(values)
    if not values:
        return {"count": 0}
    return dict(count=len(values), total=sum(values), mean=float(np.mean(values)),
                median=float(np.median(values)), p90=float(np.quantile(values, .9)),
                maximum=max(values))


def chars(value):
    return len(json.dumps(value, ensure_ascii=False))


def audit(root):
    turns, episodes, reviews, strategies = [], [], [], []
    thread_settings, counts = Counter(), Counter()
    tool_sizes, tool_seconds = defaultdict(list), defaultdict(list)
    latency = defaultdict(list)
    components = Counter()
    for episode in sorted((root / "episodes").iterdir()):
        if not (episode / "result.json").is_file():
            continue
        result = json.loads((episode / "result.json").read_text())
        turnmap, reviewmap, tool_starts = {}, {}, {}
        request_kinds, monitor_bindings = {}, set()
        episode_counts, episode_settings = Counter(), Counter()
        last_decision, last_recovery = {}, None
        for line in (episode / "events.jsonl").open():
            e = json.loads(line)
            kind = e["kind"]
            episode_counts[kind] += 1
            if kind == "codex_thread":
                setting = f"{e['model']}:{e['reasoning_effort']}"
                thread_settings[setting] += 1
                episode_settings[setting] += 1
            elif kind == "intervention_begin":
                last_decision = e["decision"]
            elif kind == "codex_turn_start":
                c = e["context"]
                turnmap[e["turn_id"]] = dict(
                    episode=episode.name, turn_id=e["turn_id"], event_id=e["id"],
                    start_ns=e["time_ns"], intent=c["intent"],
                    step=e["binding"]["observation_step"], context_chars=chars(c),
                    trigger=(c.get("decision") or last_decision).get("trigger"),
                    reason=(c.get("decision") or last_decision).get("fallback_reason"),
                    status="no_completion_event", tools=[], tool_seconds=0.,
                    tool_result_chars=0, previous_recovery=last_recovery,
                )
            elif kind == "codex_tool_start":
                tool_starts[e["call_id"]] = e["time_ns"]
                t = turnmap[e["turn_id"]]
                t.setdefault("first_tool_seconds", (e["time_ns"] - t["start_ns"]) / 1e9)
            elif kind == "codex_tool":
                t = turnmap[e["turn_id"]]
                name = e["name"]
                duration = (e["time_ns"] - tool_starts[e["call_id"]]) / 1e9
                size = chars(e["result"])
                t["tools"].append(name)
                t["tool_seconds"] += duration
                t["tool_result_chars"] += size
                tool_sizes[name].append(size)
                tool_seconds[name].append(duration)
                if name == "read_program" and e["arguments"].get("path"):
                    counts["read_program_with_path"] += 1
                if name in ("submit_intervention", "submit_review"):
                    t["proposal"] = e["arguments"].get("kind", e["arguments"].get("action"))
                    t["proposal_accepted"] = not bool(e["result"].get("error"))
                    t["proposal_event_id"] = e["id"]
            elif kind == "codex_turn_end" and e["turn_id"] in turnmap:
                t = turnmap[e["turn_id"]]
                t.update(status=e["status"], seconds=(e["time_ns"] - t["start_ns"]) / 1e9,
                         end_ns=e["time_ns"])
            elif kind == "review_begin":
                reviewmap[e["review"]] = dict(episode=episode.name, review=e["review"],
                    start_ns=e["time_ns"], trigger=e["decision"]["trigger"],
                    reason=e["decision"].get("fallback_reason"), status="unfinished")
            elif kind in ("review_end", "review_discarded", "review_cancelled", "review_failed"):
                r = reviewmap[e["review"]]
                r.update(status=kind, seconds=(e["time_ns"] - r["start_ns"]) / 1e9)
                if "result" in e:
                    r.update(action=e["result"]["action"], assessment=e["result"]["assessment"])
            elif kind == "recovery_result":
                last_recovery = {k: e.get(k) for k in
                                 ("id", "intervention_id", "verified", "error", "executed_steps")}
            elif kind == "recovery_strategy_change":
                strategies.append(dict(episode=episode.name, event_id=e["id"], plan=e["plan"]))
            elif kind == "decision_window" and e.get("backend") == "flora":
                monitor_bindings.add(e["binding"]["request_id"])
            elif kind == "flora_request":
                request_kinds[e["request_id"]] = (
                    "monitor" if e["binding"]["request_id"] in monitor_bindings else "tool")
            elif kind == "flora_result":
                r = e["result"]
                if r.get("status") == "scored" and "seconds" in r:
                    latency["flora_client_seconds"].append(r["seconds"])
                    request_kind = request_kinds.get(r["request_id"], "cached_or_unknown")
                    latency[f"flora_{request_kind}_client_seconds"].append(r["seconds"])
                    response = episode / "flora" / (r["request_id"] + ".response.json")
                    if response.is_file():
                        data = json.loads(response.read_text())
                        if "seconds" in data:
                            latency["flora_server_seconds"].append(data["seconds"])
            elif kind == "vla_inference_end" and not e.get("error"):
                latency["vla_inference_seconds"].append(e["seconds"])
        analysis_path = root / "analysis" / (episode.name + ".json")
        analysis = json.loads(analysis_path.read_text()) if analysis_path.is_file() else {}
        wall = analysis.get("wall_components_seconds", {})
        components.update(wall)
        for t in turnmap.values():
            # Never synthesize durations for cancelled/unfinished turns.
            if "seconds" in t:
                t["outside_host_tools_seconds"] = t["seconds"] - t["tool_seconds"]
        turns.extend(turnmap.values())
        reviews.extend(reviewmap.values())
        counts.update(episode_counts)
        episodes.append(dict(episode=episode.name, result=result, counts=dict(episode_counts),
                             thread_settings=dict(episode_settings), wall_components_seconds=wall))
    groups = {}
    for intent in sorted({t["intent"] for t in turns}):
        group = [t for t in turns if t["intent"] == intent]
        ended = [t for t in group if "seconds" in t]
        groups[intent] = dict(
            turns=len(group), status=dict(Counter(t["status"] for t in group)),
            proposals=dict(Counter(t.get("proposal", "none") for t in group)),
            metrics={key: stats(t[key] for t in ended if key in t) for key in
                     ("seconds", "first_tool_seconds", "tool_seconds", "outside_host_tools_seconds",
                      "context_chars", "tool_result_chars")},
        )
    report = dict(root=str(root), episodes=episodes, thread_settings=dict(thread_settings),
                  counts=dict(counts), intents=groups, turns=turns, reviews=reviews,
                  review_status=dict(Counter(r["status"] for r in reviews)),
                  review_actions=dict(Counter(r.get("action", "none") for r in reviews
                                              if r["status"] == "review_end")),
                  review_triggers=dict(Counter(r["trigger"] for r in reviews)),
                  recovery_strategy_changes=strategies,
                  wall_components_seconds=dict(components),
                  latency={k: stats(v) for k, v in latency.items()},
                  tool_result_chars={k: stats(v) for k, v in tool_sizes.items()},
                  host_tool_seconds={k: stats(v) for k, v in tool_seconds.items()},
                  notes=["Only episodes with result.json are included.",
                         "One Codex turn can contain multiple model/tool round trips.",
                         "No end event means unknown latency, not zero latency.",
                         "Background review and Flora times overlap execution.",
                         "Serialized characters are not model tokens.",
                         "Outside-host-tool time includes provider, network and reasoning; it is not GPU time."])
    (root / "agent-runtime-audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    fields = ["episode", "turn_id", "event_id", "intent", "step", "status", "seconds",
              "first_tool_seconds", "tool_seconds", "outside_host_tools_seconds", "context_chars",
              "tool_result_chars", "proposal", "proposal_accepted", "trigger", "reason"]
    with (root / "astra-turns.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(turns)
    print(json.dumps({k: report[k] for k in ("thread_settings", "intents", "review_status",
                                           "review_actions", "review_triggers", "latency",
                                           "wall_components_seconds")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    audit(parser.parse_args().root.resolve())
