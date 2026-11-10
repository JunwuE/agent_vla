"""Append-only events with lossless observations and conservative candidate export."""

from __future__ import annotations

import hashlib
import json
import random
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from .contracts import Observation


def jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dumps(value):
    return json.dumps(value, default=jsonable, ensure_ascii=False, allow_nan=False)


def save_observation_arrays(path, arrays, compression_level=6):
    """Standard lossless NPZ; compression level changes bytes on disk, never array values."""
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=compression_level, allowZip64=True) as archive:
        for name, array in arrays.items():
            with archive.open(name + '.npy', 'w', force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asanyarray(array), allow_pickle=False)


class Recorder:
    def __init__(self, root: Path, config):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "observations").mkdir()
        source_root = Path(__file__).parent
        hashes = {
            str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source_root.rglob("*"))
            if p.is_file() and p.suffix in {".py", ".md"}
        }
        (self.root / "manifest.json").write_text(
            dumps({"schema": "vla_agent.episode.v1", "config": config, "source_sha256": hashes})
        )
        self.file = (self.root / "events.jsonl").open("x", buffering=1)
        self.sequence = 0
        self.observations = {}
        self.counts = Counter()
        self.compression_level = getattr(getattr(config, 'runtime', None), 'observation_compression_level', 6)

    def event(self, kind, **payload):
        self.sequence += 1
        self.counts[kind] += 1
        row = dict(id=self.sequence, time_ns=time.time_ns(), kind=kind, **payload)
        self.file.write(dumps(row) + "\n")
        self.file.flush()
        return self.sequence

    def observe(self, obs: Observation):
        if obs.key in self.observations:
            return self.observations[obs.key]
        path = self.root / "observations" / f"{obs.step:07d}.npz"
        arrays = {"state": obs.state}
        arrays.update({f"rgb/{name}": x for name, x in obs.rgb.items()})
        arrays.update({f"depth_m/{name}": x for name, x in obs.depth_m.items()})
        save_observation_arrays(path, arrays, self.compression_level)
        record = dict(
            **obs.summary(),
            path=str(path.relative_to(self.root)),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            calibration=obs.calibration,
        )
        path.with_suffix(".json").write_text(dumps(record))
        self.event("observation", **record)
        self.observations[obs.key] = record["path"]
        return record["path"]

    def finish(self, result):
        self.event("episode_end", **result)
        (self.root / "result.json").write_text(dumps(result))

    def read_observation(self, step, episode_id, max_step, *, rgb_cameras=None, include_depth=True):
        if type(step) is not int or not 0 <= step <= max_step:
            raise ValueError("Only recorded causal observations may be read")
        key = f"{episode_id}:{step}"
        if key not in self.observations:
            raise ValueError("Observation is not recorded in this episode")
        path = self.root / self.observations[key]
        meta = json.loads(path.with_suffix(".json").read_text())
        if (
            meta["episode_id"] != episode_id
            or meta["step"] != step
            or hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]
        ):
            raise ValueError("Observation identity or checksum mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            return Observation(
                episode_id=episode_id,
                step=step,
                timestamp=meta["timestamp"],
                sim_time=meta["sim_time"],
                state=arrays["state"],
                rgb={k[4:]: arrays[k] for k in arrays.files if k.startswith("rgb/")
                     and (rgb_cameras is None or k[4:] in rgb_cameras)},
                depth_m={k[8:]: arrays[k] for k in arrays.files if include_depth and k.startswith("depth_m/")},
                calibration=meta["calibration"],
                facts=meta["facts"],
                events=tuple(meta["events"]),
                terminal=meta["terminal"],
                success=meta["success"],
                controller=meta.get("controller", {}),
                eef_control_pose=meta.get("eef_control_pose", {}),
            )

    def close(self):
        self.file.close()


def read_events(root):
    return [json.loads(line) for line in (Path(root) / "events.jsonl").read_text().splitlines()]


def export_candidates(root, output):
    """Evidence indexes, not automatically approved positive demonstrations."""
    rows = read_events(root)
    verified = {
        r["intervention_id"]: r for r in rows if r["kind"] == "intervention_end" and r.get("verified") is True
    }
    candidates = []
    for row in rows:
        proof = verified.get(row.get("intervention_id"))
        if row["kind"] == "action_executed" and proof:
            candidates.append(
                dict(
                    action_event_id=row["id"],
                    evidence_event_id=proof["id"],
                    before=row["before"],
                    after=row["after"],
                    instruction=row["instruction"],
                    source=row["source"],
                    action_layout="eef_first",
                    training_ready=False,
                    needs=["segment_review", "target_action_conversion", "frequency_alignment"],
                )
            )
    with Path(output).open("x") as handle:
        for row in candidates:
            handle.write(dumps(row) + "\n")
    return len(candidates)


def export_decision_windows(root, output, *, keep_audit_rate=0.1, seed=7):
    rows = read_events(root)
    decisions = {
        r["decision"]["binding"]["request_id"]: r["decision"] for r in rows if r["kind"] == "decision"
    }
    decision_events = {r['decision']['binding']['request_id']: r for r in rows if r['kind'] == 'decision'}
    flora_requests = {r['request_id']: r for r in rows if r['kind'] == 'flora_request'}
    rng = random.Random(seed)
    count = 0
    with Path(output).open("x") as handle:
        for row in rows:
            if row["kind"] != "decision_window":
                continue
            decision = decisions.get(row["binding"]["request_id"])
            if decision is None:
                continue
            item = dict(
                binding=row["binding"],
                observations=row["observations"],
                memory=row["memory"],
                prediction=decision,
                online_teacher_label=None,
                hindsight_outcome=None,
                review_requested=decision["label"] != "KEEP" or rng.random() < keep_audit_rate,
                split_unit="episode/task/scene",
            )
            event = decision_events[row['binding']['request_id']]
            if event.get('backend') == 'flora':
                request = flora_requests.get(event.get('model_request_id'), {})
                item.update(backend='flora', score=event.get('score'),
                            model_request_path=request.get('request_path'),
                            segment=request.get('segment'), input_kind='recorded_flora_video')
            handle.write(dumps(item) + "\n")
            count += 1
    return count


def summarize_episodes(root):
    groups = {}
    for path in Path(root).rglob("result.json"):
        manifest_path = path.with_name("manifest.json")
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != "vla_agent.episode.v1":
            continue
        config = manifest["config"]
        comparison = json.loads(dumps(config))
        comparison["environment"].pop("seed", None)
        snapshots = [r["skills"] for r in read_events(path.parent) if r["kind"] == "skill_snapshot"]
        signature = hashlib.sha256(dumps([comparison, snapshots]).encode()).hexdigest()[:12]
        group = "/".join(
            [
                config["environment"]["kind"],
                config["environment"]["task_group"] or "unassigned",
                config["environment"]["scene_split"],
                config["runtime"]["memory_mode"],
                "supervised" if config["runtime"]["supervision_enabled"] else "baseline",
                signature,
            ]
        )
        groups.setdefault(group, []).append(json.loads(path.read_text()))
    return {
        key: dict(
            episodes=len(rows),
            successes=sum(r["success"] for r in rows),
            success_rate=sum(r["success"] for r in rows) / len(rows),
            runtime_errors=sum(bool(r["error"]) for r in rows),
            astra_calls=sum(r["astra_calls"] for r in rows),
            wall_seconds=sum(r["wall_seconds"] for r in rows),
        )
        for key, rows in groups.items()
    }
