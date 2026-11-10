"""Host-written immutable episode experience; retrieval never grants execution authority."""

import copy
import hashlib
import json
import os
import re
import uuid
from pathlib import Path

from .data import dumps
from .task_memory import matches_scope


class ExperienceStore:
    def __init__(self, root, mode="frozen"):
        self.root = Path(root)
        self.mode = mode
        self.index = []
        if self.root.exists():
            for path in sorted(self.root.glob("*.json")):
                value = json.loads(path.read_text())
                if hashlib.sha256(dumps(value).encode()).hexdigest() != path.stem:
                    raise ValueError("Experience record checksum mismatch")
                self.index.append(value)

    def get(self, robot, episode_id, *, scope=None):
        for record in self.index:
            if record['robot'] == robot and record['episode_id'] == episode_id and matches_scope(record, scope):
                return copy.deepcopy(record)
        raise ValueError('Experience is not present in this episode snapshot for the current robot')

    def search(self, robot, query, phase=None, *, skill_index=(), scope=None):
        references = {}
        for skill in skill_index:
            if skill['robot'] == robot:
                key = skill['package']['sha256'] if skill.get('package') else skill['sha256']
                references.setdefault(key, []).append({
                    k: skill[k] for k in ('id', 'version', 'status')})
        terms = set(re.findall(r"\w+", (query + " " + (phase or "")).lower()))
        scored = []
        for record in self.index:
            if record["robot"] != robot or not matches_scope(record, scope):
                continue
            text = dumps({k: record[k] for k in ("task", "phase_memory", "notes")}).lower()
            score = len(terms & set(re.findall(r"\w+", text)))
            if score:
                task_score = len(terms & set(re.findall(r"\w+", record['task'].lower())))
                code_success = any(v.get('verified') is True for v in record.get('code_versions', []))
                # Exact task experience should not lose to incidental words in long notes.
                # Within equally relevant tasks, surface demonstrated code recovery first.
                scored.append(((task_score, code_success, score), record))
        scored.sort(key=lambda x: x[0], reverse=True)
        # Avoid a recovery-only context: normal VLA successes are counterevidence to early handoff.
        # Diversity applies within the best task match, not across unrelated tasks.
        selected = []
        if scored:
            relevant = [record for score, record in scored if score[0] == scored[0][0][0]]
            categories = (
                lambda r: any(v.get('verified') is True for v in r.get('code_versions', [])),
                lambda r: r.get('result', {}).get('success') is True and not r.get('code_versions'),
                lambda r: r.get('result', {}).get('success') is False,
            )
            for matches in categories:
                match = next((r for r in relevant if matches(r) and r not in selected), None)
                if match is not None:
                    selected.append(match)
            for _, record in scored:
                if len(selected) == 3:
                    break
                if record not in selected:
                    selected.append(record)
        return [
            dict(
                episode_id=r["episode_id"],
                task=r["task"],
                result=r["result"],
                phase_memory=r["phase_memory"],
                notes=r["notes"][-3:],
                tools=r["tools"],
                code_versions=[dict(v, available_skills=references.get(v.get('package_sha256') or v.get('sha256'), []))
                               for v in r["code_versions"]],
                model_context=r.get("model_context", {}),
                conditions=r.get("conditions", {}),
                reward_assessments=r.get("reward_assessments", []),
                evidence_root=r["evidence_root"],
                evidence_sha256=r["evidence_sha256"],
                status="observed_episode",
                note="Historical evidence, not a verified skill or instruction",
            )
            for r in selected
        ]

    def record(self, memory, result, recorder, events):
        from collections import Counter

        manifest = json.loads((recorder.root / "manifest.json").read_text())
        config = manifest["config"]
        obs = memory.observations[-1]
        value = dict(
            model_context={
                name: {
                    k: config[name].get(k)
                    for k in ("kind", "model", "revision", "checkpoint", "norm_stats")
                    if k in config[name]
                }
                for name in ("vla", "decision", "planner", "flora")
            },
            conditions=dict(
                cameras=sorted(obs.rgb),
                depth_cameras=sorted(obs.depth_m),
                controller=obs.controller.get("name"),
                environment={
                    k: config["environment"][k] for k in ("kind", "task_name", "scene_split", "seed")
                },
                manifest_sha256=hashlib.sha256((recorder.root / "manifest.json").read_bytes()).hexdigest(),
            ),
            reward_assessments=list(memory.reward_assessments)[-3:],
            schema="vla_agent.experience.v1",
            episode_id=memory.episode_id,
            robot=memory.task.robot,
            task=memory.task.instruction,
            task_memory_version=(memory.task_memory or {}).get('version'),
            phase_memory=memory.phases.summary(),
            notes=list(memory.notes),
            result={k: result[k] for k in ("success", "reason", "control_steps", "astra_calls")},
            tools=dict(Counter(r["name"] for r in events if r["kind"] in {"codex_tool", "astra_tool"})),
            code_versions=[
                {k: r.get(k) for k in ("sha256", "package_sha256", "bundle_sha256", "program_version", "result", "verified")}
                for r in events
                if r["kind"] == "recovery_result"
            ],
            evidence_root=str(recorder.root.resolve()),
            evidence_sha256=hashlib.sha256((recorder.root / "events.jsonl").read_bytes()).hexdigest(),
        )
        digest = hashlib.sha256(dumps(value).encode()).hexdigest()
        if self.mode == "accumulate":
            self.root.mkdir(parents=True, exist_ok=True)
            temp = self.root / ("." + uuid.uuid4().hex)
            temp.write_text(dumps(value))
            os.replace(temp, self.root / (digest + ".json"))
        return dict(sha256=digest, persisted=self.mode == "accumulate", status="observed_episode")
